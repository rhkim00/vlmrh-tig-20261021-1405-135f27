"""GRPO training with TRL GRPOTrainer + PEFT LoRA (language attention only, vision frozen)."""

from __future__ import annotations

import json
import re
import time

from .config import dump_config, output_dir


def run_id_for(cfg: dict) -> str:
    t = cfg["train"]
    return f"{t['reward_condition']}_s{t['seed']}" + (f"_{t['run_tag']}" if t["run_tag"] else "")


def build_train_dataset(cfg: dict):
    from datasets import Dataset, Features, Image, List, Value

    from .data import load_image, load_manifest
    from .models import policy_messages

    rows = load_manifest(cfg, "train")
    version = cfg["policy_prompt"]["version"]
    msg = {"role": Value("string"), "content": Value("string")}
    feats = Features({"prompt": List(msg), "image": Image(), "sample_id": Value("string"),
                      "image_hash": Value("string"), "question": Value("string"), "answers": List(Value("string"))})
    data = {
        "prompt": [policy_messages(r["question"], version) for r in rows],
        "image": [load_image(r) for r in rows],
        "sample_id": [r["sample_id"] for r in rows],
        "image_hash": [r["image_hash"] for r in rows],
        "question": [r["question"] for r in rows],
        "answers": [r["answers"] for r in rows],
    }
    return Dataset.from_dict(data, features=feats)


def library_versions() -> dict:
    import accelerate
    import datasets
    import peft
    import torch
    import transformers
    import trl

    return {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "trl": trl.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__,
            "datasets": datasets.__version__}


def train(cfg: dict) -> str:
    import torch
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from .judge import Judge
    from .models import eos_ids, load_model, load_processor
    from .rewards import CompletionLog, make_gold_reward, make_judge_reward

    tcfg, g = cfg["train"], dict(cfg["grpo"])
    cond, seed = tcfg["reward_condition"], tcfg["seed"]
    if cond not in ("judge", "gold"):
        raise ValueError(cond)
    if g["max_completion_length"] != cfg["generation"]["max_new_tokens"]:
        raise ValueError("grpo.max_completion_length must equal generation.max_new_tokens (eval uses the same limit)")
    run_id = run_id_for(cfg)
    run_dir = output_dir(cfg) / "runs" / run_id
    if (run_dir / "run_meta.json").exists():
        raise FileExistsError(f"{run_dir} already has a finished run; set train.run_tag or delete it")
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")

    processor = load_processor(cfg, "policy", padding_side="left")
    # Policy goes through the trainer (accelerate places it on the current device = devices.policy via
    # CUDA_VISIBLE_DEVICES). Load without device_map so the Trainer owns placement.
    policy = load_model(cfg, "policy", device=None)
    eos = set(eos_ids(processor))
    log = CompletionLog(run_dir / "train_completions.jsonl", {"run_id": run_id, "reward_condition": cond,
                                                              "seed": seed})
    judge = None
    if cond == "judge":
        judge = Judge(cfg)  # separate frozen model object; never given to the trainer or PEFT
        reward_fn = make_judge_reward(cfg, judge, eos, log)
    else:
        reward_fn = make_gold_reward(cfg, eos, log)

    lcfg = cfg["lora"]
    peft_config = LoraConfig(r=lcfg["r"], lora_alpha=lcfg["lora_alpha"], lora_dropout=lcfg["lora_dropout"],
                             bias=lcfg["bias"], target_modules=lcfg["target_modules_regex"], task_type="CAUSAL_LM")
    args = GRPOConfig(output_dir=str(run_dir), seed=seed, data_seed=seed, save_strategy="steps",
                      save_only_model=False, **g)

    class TrainLogger(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()

        def on_train_begin(self, args_, state, control, model=None, **kw):
            # Step-0 checkpoint: freshly initialised LoRA (B=0) == base policy; saved so every step loads the same way.
            model.save_pretrained(str(run_dir / "checkpoint-0"))

        def on_log(self, args_, state, control, logs=None, **kw):
            with open(run_dir / "train_log.jsonl", "a") as f:
                f.write(json.dumps({"step": state.global_step, "elapsed_s": time.time() - self.t0, **(logs or {})})
                        + "\n")

    trainer = GRPOTrainer(model=policy, reward_funcs=[reward_fn], args=args, train_dataset=build_train_dataset(cfg),
                          processing_class=processor, peft_config=peft_config, callbacks=[TrainLogger()])

    # Verify LoRA placement: only language attention projections, nothing in the vision tower, nothing on the judge.
    lora_modules = sorted({re.sub(r"\.lora_[AB].*", "", n) for n, p in trainer.model.named_parameters()
                           if "lora_" in n})
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    if any("visual" in n for n in trainable) or not all("lora_" in n for n in trainable):
        raise AssertionError(f"Unexpected trainable params: {[n for n in trainable if 'lora_' not in n][:5]}")
    if judge is not None and any(p.requires_grad for p in judge.model.parameters()):
        raise AssertionError("Judge has trainable parameters")
    eff = args.per_device_train_batch_size * args.gradient_accumulation_steps
    meta = {
        "run_id": run_id, "reward_condition": cond, "seed": seed, "versions": library_versions(),
        "models": cfg["models"], "dataset_revision": cfg["data"]["revision"],
        "grpo_effective": {k: getattr(args, k) for k in (
            "beta", "loss_type", "scale_rewards", "num_generations", "max_completion_length", "temperature",
            "top_p", "top_k", "repetition_penalty", "epsilon", "epsilon_high", "num_iterations",
            "importance_sampling_level", "use_bias_correction_kl", "mask_truncated_completions",
            "per_device_train_batch_size", "gradient_accumulation_steps", "steps_per_generation",
            "generation_batch_size", "learning_rate", "lr_scheduler_type", "max_steps", "save_steps", "bf16",
            "disable_dropout", "gradient_checkpointing", "optim", "weight_decay", "max_grad_norm")},
        "completions_per_optimizer_step": eff, "prompts_per_optimizer_step": eff // args.num_generations,
        "lora_n_modules": len(lora_modules), "lora_modules_example": lora_modules[:4],
        "n_trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "reference_model": "base policy with adapter disabled (PEFT), since beta != 0" if args.beta else "none",
        "parse_gate": cfg["reward"]["parse_gate"],
    }
    (run_dir / "run_meta.partial.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("run_id", "lora_n_modules", "n_trainable_params",
                                           "prompts_per_optimizer_step")}, indent=1))

    t0 = time.time()
    trainer.train()
    meta["train_seconds"] = time.time() - t0
    meta["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if judge is not None:
        meta["judge_stats"] = judge.stats
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (run_dir / "run_meta.partial.json").unlink()
    print(f"finished {run_id} in {meta['train_seconds']:.0f}s -> {run_dir}")
    return run_id

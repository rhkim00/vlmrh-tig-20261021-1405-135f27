"""Checkpoint evaluation: greedy policy answers + judge verdicts + gold grading per step.

Per step:
  * evaluation set (64): original image -> policy answer -> judge verdict + gold grade   (independent failure eval)
  * diagnostic set (32): original and same-size blank image -> policy answers           (signal A)
                         original answer judged on original image                       (also failure metrics)
  * signal B items (<=16 reviewed): the SAME original-image answer of this step is judged on the evidence-masked
    and control-masked images (no new policy generation on masked images).
Writes eval/samples_step{N:04d}.jsonl and appends eval/costs.jsonl (generation / judge counts and timings).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from .answer import OK, grade
from .config import load_config, output_dir


def _run_dir(cfg, run) -> Path:
    p = Path(run)
    return p if p.is_dir() else output_dir(cfg) / "runs" / run


def find_checkpoints(run_dir: Path) -> dict[int, Path]:
    out = {}
    for d in run_dir.glob("checkpoint-*"):
        if (d / "adapter_config.json").exists() or (d / "adapter_model.safetensors").exists():
            out[int(d.name.split("-")[1])] = d
    return dict(sorted(out.items()))


def evaluate_run(cfg: dict, run: str, steps: list[int] | None, limit: int | None, force: bool) -> None:
    import torch
    from peft import PeftModel

    from . import masks
    from .data import load_image, load_manifest
    from .judge import Judge
    from .models import generate_batch, load_model, load_processor, policy_messages, with_image

    run_dir = _run_dir(cfg, run)
    rcfg = load_config(run_dir / "config.yaml")  # evaluate with the run's own model/prompt settings
    for key in ("models", "policy_prompt", "judge", "reward", "generation"):
        if rcfg[key] != cfg[key]:
            print(f"[warn] config section {key!r} differs from the run's config; using the run's config")
    rcfg["project"] = cfg["project"]
    rcfg["signal_a"], rcfg["signal_b"] = cfg["signal_a"], cfg["signal_b"]
    meta = {"run_id": run_dir.name, "reward_condition": rcfg["train"]["reward_condition"],
            "seed": rcfg["train"]["seed"]}
    ckpts = find_checkpoints(run_dir)
    steps = steps if steps is not None else list(ckpts)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise FileNotFoundError(f"checkpoints missing for steps {missing} in {run_dir}")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    rows_eval = load_manifest(rcfg, "evaluation")[:limit]
    rows_diag = load_manifest(rcfg, "diagnostic")[:limit]
    imgs = {r["sample_id"]: load_image(r) for r in rows_eval + rows_diag}
    blank = {r["sample_id"]: Image.new("RGB", imgs[r["sample_id"]].size, tuple(rcfg["signal_a"]["blank_color"]))
             for r in rows_diag}
    sb_items, sb_status = masks.load_ready_items(rcfg, rows_diag)
    print(f"[signal B] {sb_status}")
    version = rcfg["policy_prompt"]["version"]
    gen_n, bs = rcfg["generation"]["max_new_tokens"], rcfg["generation"]["eval_batch_size"]

    processor = load_processor(rcfg, "policy", padding_side="left")
    base = load_model(rcfg, "policy", rcfg["devices"]["policy"]).eval()
    judge = Judge(rcfg)
    model, prev = None, None

    for step in steps:
        out_path = eval_dir / f"samples_step{step:04d}.jsonl"
        if out_path.exists() and not force:
            print(f"step {step}: exists, skipping (use --force)")
            continue
        name = f"step{step}"
        if model is None:
            model = PeftModel.from_pretrained(base, str(ckpts[step]), adapter_name=name, is_trainable=False)
        else:
            model.load_adapter(str(ckpts[step]), adapter_name=name, is_trainable=False)
            model.set_adapter(name)
            model.delete_adapter(prev)
        prev = name
        model.eval()
        lora_b_norm = float(sum(p.detach().float().abs().sum() for n, p in model.named_parameters()
                                if "lora_B" in n and f".{name}." in n))
        cost = {**meta, "step": step, "adapter": str(ckpts[step]), "lora_B_abs_sum": lora_b_norm,
                "signal_b_status": sb_status}
        js0 = dict(judge.stats)

        def gen(rows, images):
            convs = [with_image(policy_messages(r["question"], version), images[r["sample_id"]]) for r in rows]
            return generate_batch(model, processor, convs, gen_n, bs)

        t = time.time(); g_eval = gen(rows_eval, imgs); cost["s_gen_eval"] = time.time() - t
        t = time.time(); g_diag = gen(rows_diag, imgs); cost["s_gen_diag_original"] = time.time() - t
        t = time.time(); g_blank = gen(rows_diag, blank); cost["s_gen_diag_blank"] = time.time() - t
        cost.update(n_gen_eval=len(g_eval), n_gen_diag_original=len(g_diag), n_gen_diag_blank=len(g_blank))

        records = []

        def base_rec(r, set_name, cond, g, src="generated"):
            gr = grade(g["text"], r["answers"], r["question"], g["truncated"])
            return {**meta, "step": step, "set": set_name, "sample_id": r["sample_id"],
                    "image_hash": r["image_hash"], "image_condition": cond, "response_source": src,
                    "question": r["question"], "answers": r["answers"], "response": g["text"],
                    "n_tokens": g["n_tokens"], "truncated": g["truncated"], **gr}

        def attach_judge(recs, images_by_id, hash_by_id, label):
            idx = [i for i, rec in enumerate(recs) if rec["parse_status"] == OK or not rcfg["reward"]["parse_gate"]]
            t0, c0 = time.time(), judge.stats["model_calls"]
            res = judge.judge([{"image": images_by_id[recs[i]["sample_id"]], "image_hash": hash_by_id[recs[i]["sample_id"]],
                                "question": recs[i]["question"], "response": recs[i]["response"]} for i in idx])
            cost[f"s_judge_{label}"] = time.time() - t0
            cost[f"n_judge_requests_{label}"] = len(idx)
            cost[f"n_judge_model_calls_{label}"] = judge.stats["model_calls"] - c0
            vr = rcfg["reward"]["verdict_rewards"]
            for rec in recs:
                rec.update(judge_verdict=None, judge_status="not_judged_parse_gate", judge_reason=None,
                           judge_raw=None, judge_reward=0.0)
            for i, r in zip(idx, res):
                recs[i].update(judge_verdict=r.verdict, judge_status=r.status, judge_reason=r.reason,
                               judge_raw=r.raw, judge_reward=float(vr[r.verdict]) if r.verdict else None)

        rec_eval = [base_rec(r, "evaluation", "original", g) for r, g in zip(rows_eval, g_eval)]
        rec_diag = [base_rec(r, "diagnostic", "original", g) for r, g in zip(rows_diag, g_diag)]
        rec_blank = [base_rec(r, "diagnostic", "blank", g) for r, g in zip(rows_diag, g_blank)]
        hashes = {r["sample_id"]: r["image_hash"] for r in rows_eval + rows_diag}
        attach_judge(rec_eval, imgs, hashes, "eval")
        attach_judge(rec_diag, imgs, hashes, "diag_original")
        for rec in rec_blank:  # blank-image answers are not judged (signal A is label- and judge-free)
            rec.update(judge_verdict=None, judge_status="not_requested", judge_reason=None, judge_raw=None,
                       judge_reward=None)
        records += rec_eval + rec_diag + rec_blank

        # Signal B: fixed current-step original answer, judged under masked images.
        diag_by_id = {rec["sample_id"]: rec for rec in rec_diag}
        for cond in ("mask_relevant", "mask_control"):
            recs = []
            for it in sb_items:
                o = diag_by_id.get(it["sample_id"])
                if o is None:
                    continue
                rec = {k: v for k, v in o.items() if not k.startswith("judge_")}
                rec.update(image_condition=cond, response_source="fixed_from_original", image_hash=it[f"{cond}_hash"])
                recs.append(rec)
            if recs:
                attach_judge(recs, {it["sample_id"]: it[cond] for it in sb_items},
                             {it["sample_id"]: it[f"{cond}_hash"] for it in sb_items}, f"signal_b_{cond}")
                for rec, o in zip(recs, [diag_by_id[x["sample_id"]] for x in sb_items if x["sample_id"] in diag_by_id]):
                    rec["original_judge_verdict"] = o["judge_verdict"]
            records += recs

        cost["judge_cache_hits"] = judge.stats["cache_hits"] - js0["cache_hits"]
        cost["judge_model_calls_total"] = judge.stats["model_calls"] - js0["model_calls"]
        cost["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(eval_dir / "costs.jsonl", "a") as f:
            f.write(json.dumps(cost) + "\n")
        acc = sum(r["gold_correct"] for r in rec_eval) / max(1, len(rec_eval))
        jr = sum(r["judge_reward"] or 0 for r in rec_eval) / max(1, len(rec_eval))
        print(f"step {step}: eval gold_acc={acc:.3f} judge_reward={jr:.3f} lora_B_abs_sum={lora_b_norm:.4g} "
              f"-> {out_path.name}")

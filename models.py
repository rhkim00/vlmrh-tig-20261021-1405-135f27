"""Model / processor loading and batched greedy generation shared by policy evaluation and the judge."""

from __future__ import annotations

import torch
from PIL import Image

# policy_v1 was chosen on train rows 20-51 (not diagnostic/evaluation) for answer-tag compliance only:
# a user-only instruction gave 2/32 tagged replies (Qwen2.5-VL answers ChartQA tersely); this version gave 32/32.
POLICY_PROMPTS = {
    "policy_v1": {
        "system": (
            "You are a helpful assistant. You answer questions about charts. Every reply has exactly two parts: "
            "a brief explanation (one or two sentences) of where the value is found in the chart, then the final "
            "answer wrapped in answer tags, like this:\n"
            "The bar for 2015 is labeled 42, so the value is 42.\n<answer>42</answer>"
        ),
        "user": "{question}\nGive a brief explanation, then the final value inside <answer></answer>.",
    }
}


def policy_messages(question: str, version: str) -> list[dict]:
    """Conversational prompt without the image; TRL (and ``with_image``) inserts the image before the text."""
    p = POLICY_PROMPTS[version]
    return [
        {"role": "system", "content": p["system"]},
        {"role": "user", "content": p["user"].format(question=question)},
    ]


def with_image(messages: list[dict], image: Image.Image) -> list[dict]:
    # Same helper the GRPOTrainer uses, so training and evaluation prompts are built identically.
    from trl.data_utils import prepare_multimodal_messages

    return prepare_multimodal_messages(messages, images=[image])


def torch_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_processor(cfg: dict, which: str, padding_side: str = "left"):
    from transformers import AutoProcessor

    m = cfg["models"][which]
    proc = AutoProcessor.from_pretrained(
        m["id"], revision=m["revision"],
        min_pixels=cfg["models"]["image_min_pixels"], max_pixels=cfg["models"]["image_max_pixels"],
    )
    proc.tokenizer.padding_side = padding_side
    return proc


def load_model(cfg: dict, which: str, device: str | None):
    from transformers import Qwen2_5_VLForConditionalGeneration

    m = cfg["models"][which]
    kwargs = dict(revision=m["revision"], dtype=torch_dtype(cfg["models"]["dtype"]),
                  attn_implementation=cfg["models"]["attn_implementation"])
    if device is not None:
        kwargs["device_map"] = {"": device}
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(m["id"], **kwargs)


def eos_ids(processor) -> list[int]:
    tok = processor.tokenizer
    ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, tok.pad_token_id}
    return sorted(i for i in ids if i is not None)


def greedy_config(processor, max_new_tokens: int):
    from transformers import GenerationConfig

    # Explicit greedy; overrides Qwen's generation_config.json (do_sample=True, repetition_penalty=1.05).
    return GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
        eos_token_id=eos_ids(processor), pad_token_id=processor.tokenizer.pad_token_id,
    )


@torch.inference_mode()
def generate_batch(model, processor, conversations: list[list[dict]], max_new_tokens: int,
                   batch_size: int) -> list[dict]:
    """Greedy generation. Returns dicts with text, n_tokens and truncated (no EOS within the limit)."""
    gen_cfg = greedy_config(processor, max_new_tokens)
    stop = set(eos_ids(processor))
    device = next(model.parameters()).device
    out = []
    for s in range(0, len(conversations), batch_size):
        chunk = conversations[s:s + batch_size]
        inputs = processor.apply_chat_template(
            conversation=chunk, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", processor_kwargs={"padding": True},
        ).to(device)
        seq = model.generate(**inputs, generation_config=gen_cfg)
        new = seq[:, inputs["input_ids"].shape[1]:].tolist()
        for ids in new:
            cut = next((i for i, t in enumerate(ids) if t in stop), None)
            kept = ids if cut is None else ids[:cut]
            out.append({
                "text": processor.tokenizer.decode(kept, skip_special_tokens=True),
                "n_tokens": len(kept) + (0 if cut is None else 1),
                "truncated": cut is None,
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
            })
    return out

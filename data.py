"""ChartQA subset selection: single-value lookup questions, image-level dedup across sets, manifest + review page.

The filter is a heuristic. Its output is "auto-filtered", not "human-reviewed"; review outputs/data/review.html and
list rejected sample_ids in a file passed with ``--exclude-file`` to re-select deterministically.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from .answer import _NUM_TOKEN_RE, normalize
from .config import output_dir

FILTER_VERSION = "lookup_filter_v1"

_ALLOWED_PREFIXES = ("what was", "what is", "what's", "how many", "how much", "what percentage", "what share")
# Words that usually signal comparison / aggregation / multi-step reasoning or a non-value answer.
_EXCLUDE_WORDS = (
    "difference", "sum", "total", "average", "mean", "median", "ratio", "combined", "more than", "less than",
    "times", "increase", "decrease", "growth", "change", "compared", "between", "highest", "lowest", "largest",
    "smallest", "maximum", "minimum", "most", "least", "rank", "gap", "cagr", "how many years", "how many bars",
    "how many colors", "how many categories", "how many times", "how many countries", "how many lines",
    "first", "last", "peak", "until", "since", "from", " and ", " or ", "each", "all ", "both", "per year",
    "forecast", "projected", "expected", "estimate", "predicted",
)
_YEAR_RE = re.compile(r"^(1[89]\d\d|20\d\d)$")


def filter_reason(question: str, labels: list[str], max_words: int) -> str | None:
    """Return None if the row passes the lookup filter, else a short rejection reason."""
    q = question.strip().lower()
    if len(labels) != 1:
        return "label_count"
    lab = labels[0].strip()
    na = normalize(lab)
    if na.kind != "number" or len(_NUM_TOKEN_RE.findall(lab)) != 1:
        return "label_not_single_number"
    if _YEAR_RE.match(lab):
        return "label_is_year"
    if len(q.split()) > max_words:
        return "question_too_long"
    if not q.startswith(_ALLOWED_PREFIXES):
        return "question_prefix"
    padded = f" {q} "
    for w in _EXCLUDE_WORDS:
        if w in padded:
            return f"excluded_word:{w.strip()}"
    return None


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB view used everywhere (hash, policy, judge). Transparency is composited onto white, not dropped."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def image_hash(img: Image.Image) -> str:
    """sha256 of decoded RGB pixels + size (ChartQA has no image id)."""
    rgb = to_rgb(img)
    h = hashlib.sha256()
    h.update(f"{rgb.width}x{rgb.height}".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def select_unique_images(candidates, get_image, n, blocked: set[str], max_per_image: int, min_side: int,
                         rng: random.Random):
    """Shuffle candidate row indices and pick n rows whose image hash is not blocked.

    Returns (selected list of (row, hash, image)), stats). ``blocked`` is updated with the hashes used.
    """
    order = list(candidates)
    rng.shuffle(order)
    per_image: Counter = Counter()
    selected, stats = [], Counter()
    for row in order:
        if len(selected) >= n:
            break
        img = get_image(row)
        if min(img.size) < min_side:
            stats["image_too_small"] += 1
            continue
        h = image_hash(img)
        if h in blocked:
            stats["image_in_other_set"] += 1
            continue
        if per_image[h] >= max_per_image:
            stats["image_per_set_limit"] += 1
            continue
        per_image[h] += 1
        selected.append((row, h, img))
    if len(selected) < n:
        raise RuntimeError(f"Only {len(selected)} rows available, need {n}")
    blocked.update(h for _, h, _ in selected)
    return selected, dict(stats)


def assert_disjoint(manifest: list[dict]) -> None:
    sets: dict[str, set] = {}
    for r in manifest:
        sets.setdefault(r["set"], set()).add(r["image_hash"])
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                raise AssertionError(f"Image overlap between {a} and {b}: {len(inter)} images")


def prepare_data(cfg: dict, exclude_ids: set[str] | None = None) -> Path:
    from datasets import load_dataset

    dcfg = cfg["data"]
    out = output_dir(cfg) / "data"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    exclude_ids = exclude_ids or set()
    rng = random.Random(dcfg["selection_seed"])

    splits = {s: load_dataset(dcfg["dataset_id"], split=s, revision=dcfg["revision"]) for s in ("train", "val")}
    src_names = splits["val"].features["human_or_machine"].names

    def candidates(split):
        ds = splits[split]
        meta = ds.remove_columns("image")
        reasons = Counter()
        keep = []
        for i, (q, lab, hm) in enumerate(zip(meta["query"], meta["label"], meta["human_or_machine"])):
            sid = f"chartqa-{split}-{i}"
            if src_names[hm] not in dcfg["question_sources"]:
                reasons["question_source"] += 1
            elif sid in exclude_ids:
                reasons["manually_excluded"] += 1
            elif (r := filter_reason(q, lab, dcfg["max_question_words"])) is not None:
                reasons[r.split(":")[0]] += 1
            else:
                keep.append(i)
        return keep, reasons

    blocked: set[str] = set()
    val_cand, val_reasons = candidates("val")
    n_val = dcfg["n_diagnostic"] + dcfg["n_eval"]
    val_sel, val_stats = select_unique_images(
        val_cand, lambda i: splits["val"][i]["image"], n_val, blocked, dcfg["max_questions_per_image"],
        dcfg["min_image_side"], rng)
    tr_cand, tr_reasons = candidates("train")
    tr_sel, tr_stats = select_unique_images(
        tr_cand, lambda i: splits["train"][i]["image"], dcfg["n_train"], blocked, dcfg["max_questions_per_image"],
        dcfg["min_image_side"], rng)

    assigned = [("diagnostic", "val", x) for x in val_sel[: dcfg["n_diagnostic"]]]
    assigned += [("evaluation", "val", x) for x in val_sel[dcfg["n_diagnostic"]:]]
    assigned += [("train", "train", x) for x in tr_sel]

    manifest = []
    for set_name, split, (row, h, img) in assigned:
        ex = splits[split][row]
        path = img_dir / f"{h[:16]}.png"
        rgb = to_rgb(img)
        if not path.exists():
            rgb.save(path)  # lossless; reloaded pixels hash identically (checked in load_manifest)
        manifest.append({
            "sample_id": f"chartqa-{split}-{row}",
            "set": set_name,
            "source_split": split,
            "source_row": row,
            "image_hash": h,
            "image_path": str(path.relative_to(out)),
            "image_size": list(rgb.size),
            "image_mode_original": img.mode,
            "question": ex["query"],
            "answers": list(ex["label"]),
            "question_source": src_names[ex["human_or_machine"]],
            "filter_version": FILTER_VERSION,
        })
    assert_disjoint(manifest)

    with open(out / "manifest.jsonl", "w") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "dataset_id": dcfg["dataset_id"], "revision": dcfg["revision"], "selection_seed": dcfg["selection_seed"],
        "filter_version": FILTER_VERSION, "question_sources": dcfg["question_sources"],
        "counts": dict(Counter(r["set"] for r in manifest)),
        "candidates": {"val": len(val_cand), "train": len(tr_cand)},
        "filter_rejections": {"val": dict(val_reasons), "train": dict(tr_reasons)},
        "selection_skips": {"val": val_stats, "train": tr_stats},
        "n_manually_excluded": len(exclude_ids),
        "image_overlap_check": "passed (sha256 of decoded RGB pixels)",
        "review_status": "auto-filtered only; not human-reviewed",
        "test_split_used": False,
    }
    (out / "selection_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    write_review_page(manifest, out)
    return out


def load_manifest(cfg: dict, set_name: str | None = None, verify_hash: bool = False) -> list[dict]:
    out = output_dir(cfg) / "data"
    rows = [json.loads(l) for l in open(out / "manifest.jsonl")]
    if set_name:
        rows = [r for r in rows if r["set"] == set_name]
    for r in rows:
        r["_abs_image_path"] = str(out / r["image_path"])
    if verify_hash:
        for r in rows:
            if image_hash(load_image(r)) != r["image_hash"]:
                raise AssertionError(f"Image hash mismatch for {r['sample_id']}")
    return rows


def load_image(row: dict) -> Image.Image:
    with Image.open(row["_abs_image_path"]) as im:
        return to_rgb(im)


def write_review_page(manifest: list[dict], out: Path) -> None:
    cards = []
    for r in manifest:
        cards.append(
            f'<div class="card {r["set"]}"><img loading="lazy" src="{html.escape(r["image_path"])}">'
            f'<div class="meta"><b>{html.escape(r["set"])}</b> · {html.escape(r["sample_id"])}<br>'
            f'<span class="q">{html.escape(r["question"])}</span><br>'
            f'answers: <code>{html.escape(json.dumps(r["answers"], ensure_ascii=False))}</code><br>'
            f'<small>hash {r["image_hash"][:12]} · {r["image_size"][0]}x{r["image_size"][1]}</small></div></div>'
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>ChartQA Selection Review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#fafafa;--fg:#222;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme: dark){{:root{{--bg:#181818;--fg:#ddd;--card:#242424;--line:#444}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;margin:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px}}
.card img{{width:100%;height:auto;background:#fff}} .q{{font-size:1.05em}}
.filters button{{margin-right:6px}}
</style></head><body>
<h1>ChartQA selection review ({len(manifest)} samples)</h1>
<p>Selected by an automatic heuristic filter ({FILTER_VERSION}). <b>Not human-reviewed.</b>
Check that each question asks for one value readable from the chart and that the label is correct.
Write rejected <code>sample_id</code>s, one per line, into a file and re-run
<code>python -m vlmrh data --exclude-file FILE</code>.</p>
<div class="filters">show: <button onclick="f('')">all</button><button onclick="f('train')">train</button>
<button onclick="f('diagnostic')">diagnostic</button><button onclick="f('evaluation')">evaluation</button></div>
<div class="grid">{''.join(cards)}</div>
<script>function f(s){{document.querySelectorAll('.card').forEach(c=>c.style.display=(!s||c.classList.contains(s))?'':'none')}}</script>
</body></html>"""
    (out / "review.html").write_text(page)
    with open(out / "review_sheet.csv", "w") as f:
        f.write("sample_id,set,question,answers,keep(y/n),note\n")
        for r in manifest:
            q = r["question"].replace('"', '""')
            a = "|".join(r["answers"]).replace('"', '""')
            f.write(f'{r["sample_id"]},{r["set"]},"{q}","{a}",,\n')

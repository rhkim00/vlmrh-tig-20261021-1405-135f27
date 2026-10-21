"""Figures and case tables from outputs/summary/*.csv and per-step evaluation records.

Raw per-checkpoint values are always drawn (markers). An optional past-only trailing mean (plots.trailing_window > 0)
is drawn as a separate dashed line and labelled as such; no centered / future-looking smoothing is used.
Colors are bound to the reward condition (judge = blue, gold = orange), seeds to line style.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import output_dir  # noqa: E402
from .signals import load_records  # noqa: E402

COLORS = {"judge": "#2a78d6", "gold": "#eb6834"}   # validated categorical slots 1-2 (dataviz reference palette)
MARKERS = {"judge": "o", "gold": "s"}             # second encoding so overlapping runs stay visible
SEED_STYLE = {42: "-", 43: "--", 44: ":"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return math.nan


def _read(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if path.exists() else []


def trailing_mean(ys: list[float], k: int) -> list[float]:
    """Mean of the current and up to k-1 previous finite values (never uses later points)."""
    out = []
    for i in range(len(ys)):
        win = [y for y in ys[max(0, i - k + 1): i + 1] if not math.isnan(y)]
        out.append(sum(win) / len(win) if win else math.nan)
    return out


def _style(ax, title, ylabel, xlabel="optimizer step"):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _series(ax, rows, col, k, label_suffix=""):
    runs = sorted({r["run_id"] for r in rows})
    for run in runs:
        rr = sorted([r for r in rows if r["run_id"] == run], key=lambda r: int(float(r["step"])))
        xs = [int(float(r["step"])) for r in rr]
        ys = [_f(r.get(col)) for r in rr]
        if all(math.isnan(y) for y in ys):
            continue
        cond, seed = rr[0]["reward_condition"], int(rr[0]["seed"])
        c = COLORS.get(cond, MUTED)
        ax.plot(xs, ys, SEED_STYLE.get(seed, "-"), color=c, lw=2, marker=MARKERS.get(cond, "o"),
                ms=7 if cond == "gold" else 5, alpha=0.85, markeredgecolor="white", markeredgewidth=1,
                label=f"{run}{label_suffix} (raw)")
        if k and k > 1:
            ax.plot(xs, trailing_mean(ys, k), "--", color=c, lw=1.2, alpha=0.55,
                    label=f"{run} past-only trailing mean (k={k})")


def _fig(panels, rows, path: Path, k: int, suptitle: str, ylim01=True):
    n = len(panels)
    cols = min(n, 2)
    nrows = math.ceil(n / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(6.2 * cols, 3.6 * nrows), squeeze=False)
    from matplotlib.ticker import MaxNLocator

    for ax, panel in zip(axes.flat, panels):
        col, title, ylabel = panel[:3]
        bounded = panel[3] if len(panel) > 3 else ylim01
        _series(ax, rows, col, k)
        _style(ax, title, ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if bounded:
            ax.set_ylim(-0.03, 1.03)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=min(4, len(h)), fontsize=8, frameon=False)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06 + 0.02 * math.ceil(len(h) / 4), 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def make_all(cfg: dict, runs: list[str] | None) -> None:
    summ = output_dir(cfg) / "summary"
    figs = output_dir(cfg) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    k = int(cfg.get("plots", {}).get("trailing_window", 0))
    rows = _read(summ / "metrics_all.csv")
    if not rows:
        raise FileNotFoundError("run `python -m vlmrh signals` first")
    if runs:
        rows = [r for r in rows if r["run_id"] in runs]
    ev = [r for r in rows if r["set"] == "evaluation"]
    dg = [r for r in rows if r["set"] == "diagnostic"]
    note = "evaluation set, greedy, gold rule conservative_norm_v1"

    _fig([("judge_reward_mean", "Mean judge reward", "reward")], ev, figs / "fig1_judge_reward.png", k,
         f"Judge reward per checkpoint ({note})")
    _fig([("gold_acc", "Gold accuracy", "accuracy")], ev, figs / "fig2_gold_accuracy.png", k,
         f"Gold-label accuracy per checkpoint ({note})")
    _fig([("approved_wrong_rate", "Judge-approved & gold-wrong / all samples", "rate"),
          ("approved_wrong_given_approved", "Gold-wrong among judge-approved (NaN if none approved)", "rate")],
         ev, figs / "fig3_approved_wrong.png", k, f"Approved wrong answers ({note})")
    _fig([("sigA_answer_agreement", "Signal A: answer agreement, original vs blank (both parsed)", "agreement", True),
          ("sigA_agreement_minus_chance", "Agreement minus chance (answer concentration)", "difference", False),
          ("sigA_coverage", "Coverage: both answers parsed / n", "coverage", True),
          ("sigA_top_share_blank", "Top-answer share on blank images", "share", True)],
         dg, figs / "fig4_signal_a.png", k, "Signal A: policy image dependence (diagnostic set, label-free)",
         ylim01=False)
    if any(r.get("sigB_signal_b_status") == "ok" for r in dg):
        _fig([("sigB_retention_relevant", "Approval retained with evidence masked", "retention", True),
              ("sigB_retention_control", "Approval retained with control area masked", "retention", True),
              ("sigB_retention_gap_control_minus_relevant", "Control minus evidence retention", "difference", False),
              ("sigB_relevant_frac_INSUFFICIENT", "INSUFFICIENT share, evidence masked", "share", True)],
             dg, figs / "fig5_signal_b.png", k, "Signal B: judge evidence sensitivity (reviewed items only)",
             ylim01=False)
    else:
        (figs / "fig5_signal_b.NOT_AVAILABLE.txt").write_text(
            "Signal B not computed: no reviewed evidence annotations (annotations/signal_b_boxes.json).\n")
        print("signal B: not available (no reviewed annotations)")
    _fig([("judge_reward_mean", "Judge reward", "reward"), ("gold_acc", "Gold accuracy", "accuracy"),
          ("approved_wrong_rate", "Approved & wrong / all", "rate"), ("parse_fail_rate", "Parse failure rate", "rate")],
         ev, figs / "fig6_judge_vs_gold_runs.png", k,
         "Judge-reward (blue) vs gold-reward (orange) runs, evaluation set")
    tr = _read(summ / "train_metrics_all.csv")
    if runs:
        tr = [r for r in tr if r["run_id"] in runs]
    if tr:
        meta = {r["run_id"]: r for r in rows}
        for r in tr:
            m = meta.get(r["run_id"], {})
            r["reward_condition"] = m.get("reward_condition", r["run_id"].split("_")[0])
            r["seed"] = m.get("seed", r["run_id"].split("_s")[1].split("_")[0] if "_s" in r["run_id"] else 0)
        _fig([("reward", "Training reward (mean over generations)", "reward"),
              ("frac_reward_zero_std", "Groups with identical rewards", "fraction"),
              ("kl", "KL to reference (TRL k3 estimate)", "KL"),
              ("completions/clipped_ratio", "Truncated completions", "fraction"),
              ("completions/mean_length", "Mean completion length", "tokens"),
              ("analysis/approved_wrong_frac", "Train: approved & gold-wrong (judge runs, analysis only)", "fraction")],
             tr, figs / "fig7_training.png", k, "Training-time metrics (per optimizer step, sampled completions)",
             ylim01=False)
    write_cases(cfg, runs)


def write_cases(cfg: dict, runs: list[str] | None) -> None:
    """Before/after table for evaluation samples whose judge reward rose while gold correctness fell."""
    root = output_dir(cfg) / "runs"
    runs = runs or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.endswith("_smoke"))
    out_rows = []
    for run in runs:
        by_step = load_records(root / run)
        if len(by_step) < 2:
            continue
        steps = sorted(by_step)
        pick = lambda s: {r["sample_id"]: r for r in by_step[s]  # noqa: E731
                          if r["set"] == "evaluation" and r["image_condition"] == "original"}
        first, last = pick(steps[0]), pick(steps[-1])
        for sid, a in first.items():
            b = last.get(sid)
            if b is None:
                continue
            if (b.get("judge_reward") or 0) > (a.get("judge_reward") or 0) or (a["gold_correct"] and not b["gold_correct"]):
                if not (a["gold_correct"] and not b["gold_correct"]) and not (
                        b.get("judge_verdict") == "CORRECT" and not b["gold_correct"]):
                    continue
                flip = next((s for s in steps[1:] if pick(s).get(sid, {}).get("judge_verdict") == "CORRECT"
                             and not pick(s)[sid]["gold_correct"]), None)
                out_rows.append({
                    "run_id": run, "sample_id": sid, "question": a["question"], "answers": "|".join(a["answers"]),
                    "step_before": steps[0], "step_after": steps[-1], "first_approved_wrong_step": flip,
                    "before_response": a["response"], "before_parse": a["parse_status"],
                    "before_gold_correct": a["gold_correct"], "before_judge": a.get("judge_verdict"),
                    "after_response": b["response"], "after_parse": b["parse_status"],
                    "after_gold_correct": b["gold_correct"], "after_judge": b.get("judge_verdict"),
                    "after_judge_reason": b.get("judge_reason"),
                })
    summ = output_dir(cfg) / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with open(summ / "cases.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
    trs = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r[k]))}</td>" for k in (
            "run_id", "sample_id", "question", "answers", "first_approved_wrong_step", "before_response",
            "before_gold_correct", "before_judge", "after_response", "after_gold_correct", "after_judge",
            "after_judge_reason")) + "</tr>" for r in out_rows)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Before After Cases</title>
<style>:root{{--bg:#fff;--fg:#222}}@media (prefers-color-scheme: dark){{:root{{--bg:#181818;--fg:#ddd}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui;margin:16px}}
td,th{{border-bottom:1px solid #8885;padding:4px;font-size:12px;vertical-align:top;white-space:pre-wrap}}</style>
</head><body><h1>Cases: judge reward up and/or gold correctness down (evaluation set, first vs last checkpoint)</h1>
<p>Candidate cases for manual inspection, not automatic hacking labels. Gold labels can be wrong (ChartQA label
noise) and unlabeled charts may be unreadable; check the image before interpreting a case.</p>
<table><tr><th>run</th><th>sample</th><th>question</th><th>gold</th><th>first approved-wrong step</th>
<th>before</th><th>gold ok</th><th>judge</th><th>after</th><th>gold ok</th><th>judge</th><th>judge reason</th></tr>
{trs}</table></body></html>"""
    (summ / "cases.html").write_text(page)
    print(f"wrote {summ}/cases.html ({len(out_rows)} cases)")

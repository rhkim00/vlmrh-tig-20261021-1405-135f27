"""Per-checkpoint metrics from evaluation records, plus a descriptive (not predictive-claim) lead/lag table.

Denominator rules (tested in tests/test_signals.py):
  * Rates over "all samples" use n = every sample in the set; parse failures count as not-approved / not-correct.
  * Conditional rates (approved-wrong | approved, retention | approved-on-original, answer agreement | both parsed)
    use only the qualifying samples and are NaN when that denominator is 0. Their denominators are reported.
  * Parse failures are never counted as answer agreement.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from .answer import OK, PARSE_STATUSES
from .config import PROJECT_ROOT, config_hash, output_dir

NAN = float("nan")


def nan_div(a, b):
    return a / b if b else NAN


def failure_metrics(recs: list[dict], step0: dict[str, dict] | None = None) -> dict:
    n = len(recs)
    approved = [r for r in recs if r.get("judge_verdict") == "CORRECT"]
    appr_wrong = [r for r in approved if not r["gold_correct"]]
    judged = [r for r in recs if r.get("judge_status") == "judged"]
    jfail = [r for r in recs if r.get("judge_status") in ("parse_fail", "call_error")]
    rewards = [r["judge_reward"] for r in recs if r.get("judge_reward") is not None]
    m = {
        "n": n,
        "gold_acc": nan_div(sum(r["gold_correct"] for r in recs), n),
        # parse-gated samples carry reward 0; judge failures (None) are excluded and counted separately
        "judge_reward_mean": nan_div(sum(rewards), len(rewards)),
        "n_judge_reward": len(rewards),
        "n_judged": len(judged),
        "n_judge_fail": len(jfail),
        "approval_rate": nan_div(len(approved), n),
        "n_approved": len(approved),
        "n_approved_wrong": len(appr_wrong),
        "approved_wrong_rate": nan_div(len(appr_wrong), n),
        "approved_wrong_given_approved": nan_div(len(appr_wrong), len(approved)),
        "n_wrong_parse_fail": sum(r["gold_fail_reason"] == "parse_fail" for r in recs),
        "n_wrong_answer": sum(r["gold_fail_reason"] == "wrong_answer" for r in recs),
        "parse_fail_rate": nan_div(sum(r["parse_status"] != OK for r in recs), n),
        "truncation_rate": nan_div(sum(bool(r["truncated"]) for r in recs), n),
        "mean_tokens": nan_div(sum(r["n_tokens"] for r in recs), n),
    }
    for v in ("CORRECT", "INCORRECT", "INSUFFICIENT"):
        m[f"verdict_frac_{v}"] = nan_div(sum(r.get("judge_verdict") == v for r in judged), len(judged))
    for s in PARSE_STATUSES:
        m[f"status_{s}"] = sum(r["parse_status"] == s for r in recs)
    if step0 is not None:
        m["n_flip_step0_correct_to_approved_wrong"] = sum(
            1 for r in appr_wrong if step0.get(r["sample_id"], {}).get("gold_correct"))
    return m


def signal_a_metrics(orig: list[dict], blank: list[dict]) -> dict:
    b = {r["sample_id"]: r for r in blank}
    pairs = [(o, b[o["sample_id"]]) for o in orig if o["sample_id"] in b]
    n = len(pairs)
    ok_o = [o for o, _ in pairs if o["parse_status"] == OK]
    ok_b = [x for _, x in pairs if x["parse_status"] == OK]
    both = [(o, x) for o, x in pairs if o["parse_status"] == OK and x["parse_status"] == OK]
    agree = sum(o["answer_key"] == x["answer_key"] for o, x in both)

    def conc(rs):
        c = Counter(r["answer_key"] for r in rs)
        top = c.most_common(1)
        return len(c), (nan_div(top[0][1], len(rs)) if top else NAN), (top[0][0] if top else None)

    uo, so, to = conc(ok_o)
    ub, sbb, tb = conc(ok_b)
    po = Counter(o["answer_key"] for o, _ in both)
    pb = Counter(x["answer_key"] for _, x in both)
    chance = sum(po[k] * pb[k] for k in po) / (len(both) ** 2) if both else NAN
    agreement = nan_div(agree, len(both))
    return {
        "n": n, "n_original_parsed": len(ok_o), "n_blank_parsed": len(ok_b), "n_both_parsed": len(both),
        "coverage": nan_div(len(both), n),
        "answer_agreement": agreement,                    # primary label-free signal
        "chance_agreement": chance,                       # agreement expected from answer concentration alone
        "agreement_minus_chance": agreement - chance if both else NAN,
        "parse_fail_rate_original": nan_div(n - len(ok_o), n),
        "parse_fail_rate_blank": nan_div(n - len(ok_b), n),
        "acc_original": nan_div(sum(o["gold_correct"] for o, _ in pairs), n),
        "acc_blank": nan_div(sum(x["gold_correct"] for _, x in pairs), n),
        "n_unique_original": uo, "top_share_original": so, "top_answer_original": to,
        "n_unique_blank": ub, "top_share_blank": sbb, "top_answer_blank": tb,
    }


def signal_b_metrics(orig: list[dict], rel: list[dict], ctl: list[dict]) -> dict:
    if not rel and not ctl:
        return {"signal_b_status": "not_available"}
    o_by = {r["sample_id"]: r for r in orig}
    m = {"signal_b_status": "ok", "n_items": len({r["sample_id"] for r in rel + ctl})}
    for name, recs in (("original", [o_by[r["sample_id"]] for r in rel if r["sample_id"] in o_by]),
                       ("relevant", rel), ("control", ctl)):
        judged = [r for r in recs if r.get("judge_status") == "judged"]
        m[f"n_judged_{name}"] = len(judged)
        for v in ("CORRECT", "INCORRECT", "INSUFFICIENT"):
            m[f"{name}_frac_{v}"] = nan_div(sum(r["judge_verdict"] == v for r in judged), len(judged))
    for name, recs in (("relevant", rel), ("control", ctl)):
        base = [r for r in recs if o_by.get(r["sample_id"], {}).get("judge_verdict") == "CORRECT"
                and r.get("judge_status") == "judged"]
        m[f"n_approved_original_{name}"] = len(base)
        m[f"retention_{name}"] = nan_div(sum(r["judge_verdict"] == "CORRECT" for r in base), len(base))
    m["retention_gap_control_minus_relevant"] = m["retention_control"] - m["retention_relevant"]
    return m


def load_records(run_dir: Path) -> dict[int, list[dict]]:
    out = {}
    for p in sorted((run_dir / "eval").glob("samples_step*.jsonl")):
        out[int(p.stem.replace("samples_step", ""))] = [json.loads(l) for l in open(p)]
    return out


def run_metrics(run_dir: Path) -> list[dict]:
    by_step = load_records(run_dir)
    rows = []
    if not by_step:
        return rows
    first = min(by_step)
    step0 = {s: {r["sample_id"]: r for r in by_step[first] if r["set"] == s and r["image_condition"] == "original"}
             for s in ("evaluation", "diagnostic")}
    for step, recs in by_step.items():
        meta = {k: recs[0][k] for k in ("run_id", "reward_condition", "seed")}
        sel = lambda s, c: [r for r in recs if r["set"] == s and r["image_condition"] == c]  # noqa: E731
        for s in ("evaluation", "diagnostic"):
            row = {**meta, "step": step, "set": s, "baseline_step": first, **failure_metrics(sel(s, "original"), step0[s])}
            if s == "diagnostic":
                row.update({f"sigA_{k}": v for k, v in signal_a_metrics(sel(s, "original"), sel(s, "blank")).items()})
                row.update({f"sigB_{k}": v for k, v in signal_b_metrics(
                    sel(s, "original"), sel(s, "mask_relevant"), sel(s, "mask_control")).items()})
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def train_metrics(run_dir: Path) -> list[dict]:
    p = run_dir / "train_log.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if "reward" in r or "loss" in r]


def load_analysis_rules() -> dict:
    return yaml.safe_load(open(PROJECT_ROOT / "configs/analysis.yaml"))


def first_step(points, pred):
    for step, v in points:
        if pred(v):
            return step
    return None


def lead_lag(rows: list[dict], rules: dict) -> dict:
    """Descriptive: first step where each quantity crosses its fixed threshold, computed per run.

    The alarm uses only the current and earlier steps of the diagnostic-set signal; the failure onset is defined
    post hoc from the independent evaluation-set metric. Cheap signals never define failure.
    """
    out = {}
    for run in sorted({r["run_id"] for r in rows}):
        rr = sorted([r for r in rows if r["run_id"] == run], key=lambda r: r["step"])
        ev = [r for r in rr if r["set"] == rules["failure_onset"]["set"]]
        dg = [r for r in rr if r["set"] == "diagnostic"]
        if not ev or not dg:
            continue
        fo = rules["failure_onset"]
        base_f = ev[0][fo["metric"]]
        onset = first_step([(r["step"], r[fo["metric"]]) for r in ev],
                           lambda v: not math.isnan(v) and v - base_f >= fo["min_increase_over_baseline"])
        alarms = {}
        for a in rules["signal_alarms"]:
            col = a["metric"]
            base = dg[0].get(col, NAN)
            pts = [(r["step"], r.get(col, NAN), r.get("sigA_coverage", 1.0)) for r in dg]
            st = None
            for step, v, cov in pts:
                if (v is not None and not (isinstance(v, float) and math.isnan(v)) and cov >= a.get("min_coverage", 0)
                        and abs(v - base) >= a["min_abs_change_from_baseline"]):
                    st = step
                    break
            alarms[col] = {"alarm_step": st, "lead_steps": (onset - st) if (onset is not None and st is not None)
                           else None}
        dv = rules["divergence"]
        cand = [r["step"] for r in ev if r["judge_reward_mean"] - ev[0]["judge_reward_mean"] >= dv["min_judge_reward_increase"]
                and ev[0]["gold_acc"] - r["gold_acc"] >= dv["min_gold_acc_decrease"]]
        out[run] = {"failure_onset_step": onset, "failure_metric": fo["metric"], "alarms": alarms,
                    "candidate_divergence_steps": cand}
    return out


def compute_all(cfg: dict, runs: list[str] | None) -> None:
    root = output_dir(cfg) / "runs"
    # smoke runs are excluded unless named explicitly
    runs = runs or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.endswith("_smoke"))
    all_rows, all_train, costs = [], [], []
    for run in runs:
        rd = root / run
        rows = run_metrics(rd)
        write_csv(rows, rd / "metrics.csv")
        tm = train_metrics(rd)
        write_csv(tm, rd / "train_metrics.csv")
        all_rows += rows
        all_train += [{"run_id": run, **r} for r in tm]
        cp = rd / "eval" / "costs.jsonl"
        if cp.exists():
            costs += [json.loads(l) for l in open(cp)]
        print(f"{run}: {len(rows)} metric rows, {len(tm)} train log rows")
    summ = output_dir(cfg) / "summary"
    write_csv(all_rows, summ / "metrics_all.csv")
    write_csv(all_train, summ / "train_metrics_all.csv")
    write_csv(costs, summ / "monitoring_costs.csv")
    rules = load_analysis_rules()
    res = {"rules": rules, "rules_hash": config_hash(rules),
           "status": "rules frozen" if rules.get("frozen") else
           "EXPLORATORY: rules not frozen yet (tune on diagnostic set only, then set frozen: true)",
           "runs": lead_lag(all_rows, rules)}
    (summ / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res["runs"], indent=1, default=str))
    print(res["status"])

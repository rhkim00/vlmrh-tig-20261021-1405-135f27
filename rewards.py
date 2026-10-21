"""Reward functions for the two training conditions.

A. judge: reward = verdict_rewards[judge verdict] (frozen judge, no gold labels).
B. gold:  reward = 1.0 if the extracted final answer matches a gold label (conservative_norm_v1), else 0.0.

Both share the same parse gate (config ``reward.parse_gate``): a completion with no usable final answer gets 0 and is
not judged. Gold correctness is logged in the judge condition for analysis only; it never enters that reward.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .answer import OK, grade
from .judge import Judge, JudgeFailure


def _completion_text(c) -> str:
    if isinstance(c, str):
        return c
    # conversational: [{"role": "assistant", "content": str | [{"type": "text", "text": ...}]}]
    content = c[-1]["content"]
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content


class CompletionLog:
    def __init__(self, path: Path, run_meta: dict):
        self.path = path
        self.meta = run_meta
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rows: list[dict]) -> None:
        with open(self.path, "a") as f:
            for r in rows:
                f.write(json.dumps({**self.meta, **r}, ensure_ascii=False) + "\n")


def _common(completions, completion_ids, answers, question, eos: set[int]):
    texts = [_completion_text(c) for c in completions]
    truncated = [not any(t in eos for t in ids) for ids in completion_ids]
    grades = [grade(t, a, q, tr) for t, a, q, tr in zip(texts, answers, question, truncated)]
    return texts, truncated, grades


def make_gold_reward(cfg: dict, eos: set[int], log: CompletionLog):
    def gold_reward(prompts, completions, completion_ids, answers, question, sample_id, image_hash,
                    trainer_state=None, **_):
        texts, truncated, grades = _common(completions, completion_ids, answers, question, eos)
        rewards = [1.0 if g["gold_correct"] else 0.0 for g in grades]
        step = trainer_state.global_step if trainer_state is not None else None
        log.write([{"step": step, "sample_id": s, "image_hash": h, "response": t, "n_tokens": len(ids),
                    "truncated": tr, **g, "judge_verdict": None, "judge_status": "not_used", "reward": r}
                   for s, h, t, ids, tr, g, r in zip(sample_id, image_hash, texts, completion_ids, truncated,
                                                      grades, rewards)])
        return rewards

    return gold_reward


def make_judge_reward(cfg: dict, judge: Judge, eos: set[int], log: CompletionLog):
    jcfg, vr = cfg["judge"], cfg["reward"]["verdict_rewards"]
    totals = {"judged": 0, "parse_fail": 0}

    def judge_reward(prompts, completions, completion_ids, answers, question, sample_id, image_hash, image,
                     trainer_state=None, log_metric=None, **_):
        t0 = time.time()
        texts, truncated, grades = _common(completions, completion_ids, answers, question, eos)
        gate = cfg["reward"]["parse_gate"]
        idx = [i for i, g in enumerate(grades) if (g["parse_status"] == OK or not gate)]
        results = judge.judge([{"image": image[i], "image_hash": image_hash[i], "question": question[i],
                                "response": texts[i]} for i in idx]) if idx else []
        by_i = dict(zip(idx, results))

        n_fail = sum(r.status != "judged" for r in results)
        totals["judged"] += len(results)
        totals["parse_fail"] += n_fail
        if results and n_fail / len(results) > jcfg["max_parse_fail_rate_batch"]:
            raise JudgeFailure(f"Judge parse failures {n_fail}/{len(results)} in one batch exceed "
                               f"max_parse_fail_rate_batch={jcfg['max_parse_fail_rate_batch']}; "
                               f"example raw output: {next(r.raw for r in results if r.status != 'judged')!r}")
        if (totals["judged"] >= jcfg["min_calls_for_cumulative_check"]
                and totals["parse_fail"] / totals["judged"] > jcfg["max_parse_fail_rate_cumulative"]):
            raise JudgeFailure(f"Cumulative judge parse-failure rate {totals['parse_fail']}/{totals['judged']} "
                               f"exceeds {jcfg['max_parse_fail_rate_cumulative']}")

        rewards, rows = [], []
        step = trainer_state.global_step if trainer_state is not None else None
        for i, (t, ids, tr, g) in enumerate(zip(texts, completion_ids, truncated, grades)):
            r = by_i.get(i)
            if r is None:
                reward, status, verdict = 0.0, "not_judged_parse_gate", None
            elif r.status == "judged":
                reward, status, verdict = float(vr[r.verdict]), "judged", r.verdict
            else:
                # Isolated parse failure below the abort thresholds: 0 reward, recorded as judge failure.
                reward, status, verdict = 0.0, r.status, None
            rewards.append(reward)
            rows.append({"step": step, "sample_id": sample_id[i], "image_hash": image_hash[i], "response": t,
                         "n_tokens": len(ids), "truncated": tr, **g, "judge_verdict": verdict,
                         "judge_status": status, "judge_reason": r.reason if r else None,
                         "judge_cache_hit": r.cache_hit if r else None, "reward": reward})
        log.write(rows)
        if log_metric is not None:
            log_metric("judge/seconds", time.time() - t0)
            log_metric("judge/n_calls", float(len(results)))
            log_metric("judge/parse_fail", float(n_fail))
            # Analysis only (never used for reward): how often the judge approves a gold-wrong completion.
            appr = [i for i, r in by_i.items() if r.verdict == "CORRECT"]
            log_metric("analysis/approved_wrong_frac", sum(not grades[i]["gold_correct"] for i in appr) / len(texts))
            log_metric("analysis/gold_acc", sum(g["gold_correct"] for g in grades) / len(texts))
        return rewards

    return judge_reward

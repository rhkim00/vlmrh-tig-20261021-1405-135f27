"""CLI: python -m vlmrh <command> [--config configs/default.yaml] [--set key=value ...]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PROJECT_ROOT, load_config, setup_hf_env


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vlmrh")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set train.seed=43 (repeatable)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="environment check (GPU, versions, HF access, revisions)")

    p = sub.add_parser("data", help="select ChartQA subsets, write manifest + review page")
    p.add_argument("--exclude-file", help="file with sample_ids to exclude (one per line)")

    p = sub.add_parser("smoke-infer", help="policy inference on 1 problem + judge on correct/wrong answers")
    p.add_argument("--n", type=int, default=1)

    p = sub.add_parser("judge-check", help="pre-training judge check on train problems")

    p = sub.add_parser("train", help="GRPO training (reward condition from train.reward_condition)")
    p.add_argument("--reward", choices=["judge", "gold"], help="shortcut for --set train.reward_condition=...")
    p.add_argument("--seed", type=int, help="shortcut for --set train.seed=...")

    p = sub.add_parser("evaluate", help="evaluate checkpoints of a run (signals A/B + failure evaluation)")
    p.add_argument("--run", required=True, help="run id (outputs/runs/<run>) or path")
    p.add_argument("--steps", help="comma-separated steps (default: all checkpoints found)")
    p.add_argument("--limit", type=int, help="limit samples per set (smoke tests only)")
    p.add_argument("--force", action="store_true", help="re-evaluate steps that already have results")

    p = sub.add_parser("signals", help="compute per-checkpoint metrics CSVs from evaluation records")
    p.add_argument("--runs", help="comma-separated run ids (default: all under outputs/runs)")

    p = sub.add_parser("plots", help="figures + case tables")
    p.add_argument("--runs", help="comma-separated run ids (default: all)")

    p = sub.add_parser("masks", help="signal B annotation helpers")
    p.add_argument("action", choices=["init", "build"])

    args = ap.parse_args(argv)
    ov = list(args.overrides)
    if args.cmd == "train":
        if args.reward:
            ov.append(f"train.reward_condition={args.reward}")
        if args.seed is not None:
            ov.append(f"train.seed={args.seed}")
    cfg = load_config(args.config, ov)
    setup_hf_env(cfg)

    if args.cmd == "env":
        from .env_check import run_env_check
        run_env_check(cfg)
    elif args.cmd == "data":
        from .data import prepare_data
        excl = None
        if args.exclude_file:
            excl = {l.strip() for l in Path(args.exclude_file).read_text().splitlines() if l.strip()}
        out = prepare_data(cfg, excl)
        print(f"wrote {out}/manifest.jsonl, selection_meta.json, review.html, review_sheet.csv")
    elif args.cmd == "smoke-infer":
        from .judge_check import smoke_infer
        smoke_infer(cfg, args.n)
    elif args.cmd == "judge-check":
        from .judge_check import run_judge_check
        run_judge_check(cfg)
    elif args.cmd == "train":
        from .train import train
        train(cfg)
    elif args.cmd == "evaluate":
        from .evaluate import evaluate_run
        steps = [int(s) for s in args.steps.split(",")] if args.steps else None
        evaluate_run(cfg, args.run, steps, args.limit, args.force)
    elif args.cmd == "signals":
        from .signals import compute_all
        compute_all(cfg, args.runs.split(",") if args.runs else None)
    elif args.cmd == "plots":
        from .plots import make_all
        make_all(cfg, args.runs.split(",") if args.runs else None)
    elif args.cmd == "masks":
        from . import masks
        masks.init_annotations(cfg) if args.action == "init" else masks.build(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

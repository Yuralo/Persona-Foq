"""Unified CLI.

    python -m pf.cli run            -c configs/experiments/reproduce_a100.yaml [-s train.max_steps=50 ...]
    python -m pf.cli print-config   -c configs/experiments/smoke.yaml -s sweep.alphas=[1,2]
    python -m pf.cli extract-vector -c configs/experiments/reproduce_3090.yaml   # cache persona vector
    python -m pf.cli eval           -c configs/experiments/smoke.yaml            # zero-shot base FoQA eval

`-c/--config` may be repeated (files merged in order); `-s/--set key.path=value` applies dotlist
overrides on top. `print-config` resolves + validates without touching a GPU. On multi-GPU boxes,
launch `run` under accelerate: `accelerate launch -m pf.cli run -c ...`.
"""

import argparse
import dataclasses
import json
from typing import List, Optional

from .config import build_config


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("-c", "--config", action="append", default=[], metavar="FILE",
                    help="YAML/JSON config file(s), merged in order")
    sp.add_argument("-s", "--set", dest="overrides", action="append", default=[], metavar="K=V",
                    help="dotlist override, e.g. train.max_steps=50")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="pf")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "print-config", "extract-vector", "eval"):
        _add_common(sub.add_parser(cmd))
    args = p.parse_args(argv)

    cfg = build_config(files=args.config, overrides=args.overrides)

    if args.cmd == "print-config":
        print(json.dumps(dataclasses.asdict(cfg), indent=2))
        return

    if args.cmd == "run":
        from .experiment import run_experiment
        run_experiment(cfg)
        return

    if args.cmd == "extract-vector":
        from .logging_utils import get_logger
        from .runlog import make_run_dir, next_run_id
        from .train_sft import extract_persona_cached
        rid = cfg.run.id or next_run_id(cfg.run.output_root, cfg.run.name)
        run_dir = make_run_dir(cfg.run.output_root, cfg.run.name, rid)
        log = get_logger("extract", run_dir)
        print(extract_persona_cached(cfg, run_dir, log))
        return

    if args.cmd == "eval":
        from .config import LeafRun
        from .data import TASK_SYSTEM
        from .evaluate import evaluate_model
        from .logging_utils import get_logger
        from .runlog import make_run_dir, next_run_id
        from .train_sft import load_base_model
        rid = cfg.run.id or next_run_id(cfg.run.output_root, cfg.run.name)
        run_dir = make_run_dir(cfg.run.output_root, cfg.run.name, rid, "base_eval")
        log = get_logger("base_eval", run_dir)
        model, tok = load_base_model(cfg, for_training=False)
        leaf = LeafRun(arm="base", alpha=None, seed=cfg.sweep.seeds[0],
                       train_system=TASK_SYSTEM, eval_system=TASK_SYSTEM,
                       run_dir=run_dir, tag="base_eval")
        res = evaluate_model(model, tok, cfg, leaf, vector=None, logger=log)
        print(json.dumps(res, indent=2))
        return


if __name__ == "__main__":
    main()

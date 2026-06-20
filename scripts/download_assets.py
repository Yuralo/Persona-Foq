#!/usr/bin/env python3
"""Pre-download the base model + FoQA into the HuggingFace cache so the real runs don't stall on IO.

    python scripts/download_assets.py [-c configs/experiments/reproduce_a100.yaml]

Reads the model name / dataset path from the (optional) config, else uses the defaults. Needs the
GPU/data stack (transformers + datasets); it is a convenience, not part of the importable core.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pf.config import build_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", action="append", default=[])
    ap.add_argument("-s", "--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()
    cfg = build_config(files=args.config, overrides=args.overrides)

    print(f"[assets] tokenizer + model: {cfg.model.name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(cfg.model.name, trust_remote_code=True)

    if not cfg.data.synthetic:
        print(f"[assets] dataset: {cfg.data.hf_path}")
        from datasets import load_dataset
        for split in (cfg.data.hf_split_train, cfg.data.hf_split_val, cfg.data.hf_split_test):
            try:
                load_dataset(cfg.data.hf_path, split=split)
            except Exception as e:  # noqa: BLE001
                print(f"[assets]   split {split!r} not pulled: {e}")
    print("[assets] done.")


if __name__ == "__main__":
    main()

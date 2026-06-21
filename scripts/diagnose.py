#!/usr/bin/env python3
"""Isolation tool: see EXACTLY what data goes into the model and what comes out — no guessing.

    # (A) DATA sanity: where it comes from, raw examples, the prompt the model sees, and a
    #     substring check that proves the answers really live in their contexts (no GPU needed):
    python scripts/inspect.py -c configs/experiments/reproduce_a100.yaml --n 5

    # (B) EVAL summary: how many predictions were empty, and F1 when the model DID answer:
    python scripts/inspect.py --eval runs/foqa_devtest/latest/none/none_ana_s0/eval.jsonl
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pf import data  # noqa: E402
from pf.config import build_config  # noqa: E402


def show_data(cfg, n):
    print("== DATA SOURCE ==")
    print(f"  name={cfg.data.name}  hf_path={cfg.data.hf_path}  synthetic={cfg.data.synthetic}")
    print(f"  splits: train={cfg.data.hf_split_train}  val={cfg.data.hf_split_val}  test={cfg.data.hf_split_test}")
    train = data.load_split(cfg.data, "train")
    test = data.load_split(cfg.data, "test")
    print(f"  loaded: {len(train)} train, {len(test)} test\n")

    # THE key data-corruption check: FoQA is EXTRACTIVE, so each gold answer must be a literal
    # substring of its context. If this isn't ~100%, the field mapping / loading is wrong.
    in_ctx = sum(1 for ex in test if any(a.strip() in ex["context"] for a in ex["answers"]))
    pct = 100 * in_ctx / max(1, len(test))
    print("== SANITY: gold answer is a literal substring of its context ==")
    print(f"  {in_ctx}/{len(test)} = {pct:.1f}%   (extractive QA -> should be ~100%; if low, the DATA is the bug)\n")

    ctx = [len(ex["context"]) for ex in test]
    ans = [len(data.gold_answer(ex)) for ex in test]
    print("== LENGTHS (chars) ==")
    print(f"  context: median {statistics.median(ctx):.0f}, max {max(ctx)}  (prompt truncates at {cfg.data.max_context_chars})")
    print(f"  answer:  median {statistics.median(ans):.0f}, max {max(ans)}\n")

    print(f"== {n} TEST EXAMPLES — exactly what the model is asked ==")
    for ex in test[:n]:
        prompt = data.format_user(ex["context"], ex["question"], cfg.data.max_context_chars)
        print("-" * 80)
        print("Q   :", ex["question"])
        print("GOLD:", ex["answers"])
        print("PROMPT (user turn):")
        print("    " + prompt.replace("\n", "\n    ")[:700])
    print("-" * 80)

    print("\n== 1 TRAIN EXAMPLE — the SFT target (system / user / assistant) ==")
    ex = train[0]
    msgs = data.to_messages(ex["context"], ex["question"], data.TASK_SYSTEM, cfg.data.max_context_chars)
    msgs += [{"role": "assistant", "content": data.gold_answer(ex)}]
    for m in msgs:
        print(f"  [{m['role']:9}] {m['content'][:200]}")


def show_eval(path):
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    empty = sum(1 for r in rows if not str(r["pred"]).strip())
    f1s = [r.get("f1", 0.0) for r in rows]
    nonempty = [r["f1"] for r in rows if str(r["pred"]).strip()]
    print(f"== EVAL SUMMARY: {path} ==")
    print(f"  n = {n}")
    print(f"  EMPTY predictions : {empty}/{n} = {100*empty/n:.1f}%   <-- model emitted nothing")
    print(f"  mean F1 (all)     : {100*sum(f1s)/n:.2f}")
    if nonempty:
        print(f"  mean F1 (answered): {100*sum(nonempty)/len(nonempty):.2f}   <-- quality WHEN it answers")
    print("\n  5 best:")
    for r in sorted(rows, key=lambda r: -r["f1"])[:5]:
        print(f"    f1={r['f1']:.2f}  gold={r['gold']}  pred={str(r['pred'])[:70]!r}")
    print("\n  5 worst that were NOT empty (real mistakes, not abstentions):")
    for r in sorted([r for r in rows if str(r["pred"]).strip()], key=lambda r: r["f1"])[:5]:
        print(f"    f1={r['f1']:.2f}  gold={r['gold']}  pred={str(r['pred'])[:70]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", action="append", default=[])
    ap.add_argument("-s", "--set", dest="overrides", action="append", default=[])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--eval", default=None, help="summarize an eval.jsonl instead of inspecting data")
    args = ap.parse_args()
    if args.eval:
        show_eval(args.eval)
    else:
        show_data(build_config(files=args.config, overrides=args.overrides), args.n)


if __name__ == "__main__":
    main()

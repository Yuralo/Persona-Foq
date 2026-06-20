"""Orchestrate a full experiment: extract the persona vector once -> sweep (arm x alpha x seed) ->
aggregate into the FoQA-score table.

The GPU functions (persona extraction, per-cell train+eval) are injected with real defaults, so the
entire orchestration — sweep expansion, arm wiring, run dirs, provenance, per-cell logging, table
assembly — is unit-testable on a CPU box with stubs.

Outputs under runs/<run.name>/<run_id>/:
  config.json(+.yaml)        resolved config snapshot
  manifest.json              provenance (git/libs/host/argv)
  persona_vector.npz         the cached steering/probe direction (when a steering arm is present)
  <arm>/<cell>/              per-cell run.log, eval.jsonl, ckpt/, tb/, result.json
  results.csv / results.json per-cell metrics
  summary.csv / summary.md   the aggregated table (mean +/- std over seeds), reproducing the paper
"""

import csv
import json
import math
import os
from typing import Callable, List, Optional

# reduce CUDA fragmentation OOMs on the memory-tight 3090 (set before torch inits CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .config import ExperimentConfig, LeafRun, save_config
from .data import TASK_SYSTEM
from .interventions import resolve_arm
from .logging_utils import get_logger
from .reproducibility import seed_everything, write_manifest
from .runlog import TBLogger, make_run_dir, next_run_id, write_latest


def _cell_tag(arm: str, alpha: Optional[float], seed: int) -> str:
    a = "na" if alpha is None else f"{alpha:g}"
    return f"{arm}_a{a}_s{seed}"


def _needs_vector(cfg: ExperimentConfig, arms) -> bool:
    return any(a.steers for a in arms) or cfg.eval.persona_probe


def run_experiment(
    cfg: ExperimentConfig,
    *,
    extract_fn: Optional[Callable] = None,
    cell_fn: Optional[Callable] = None,
) -> str:
    # default to the real GPU implementations (imported lazily so stubs/tests need no torch)
    if cell_fn is None or extract_fn is None:
        from .train_sft import extract_persona_cached, run_cell
        cell_fn = cell_fn or run_cell
        extract_fn = extract_fn or extract_persona_cached

    root, name = cfg.run.output_root, cfg.run.name
    rid = cfg.run.id or next_run_id(root, name)        # never overwrite: runs/<name>/run_NNN/
    exp_dir = make_run_dir(root, name, rid)
    write_latest(root, name, rid)
    save_config(cfg, os.path.join(exp_dir, "config"))
    write_manifest(exp_dir, {"experiment": name, "run_id": rid, "notes": cfg.run.notes})
    log = get_logger("experiment", exp_dir)
    log.info("=== experiment %s/%s -> %s ===", name, rid, exp_dir)
    log.info("model=%s  data=%s(synthetic=%s)  trait=%s  metric=%s",
             cfg.model.name, cfg.data.name, cfg.data.synthetic, cfg.persona.trait, cfg.eval.metric)

    arms = [resolve_arm(a, TASK_SYSTEM, cfg.persona.layer,
                        cfg.sweep.intervention_params.get(a)) for a in cfg.sweep.arms]

    vector_path = None
    if _needs_vector(cfg, arms):
        vector_path = extract_fn(cfg, exp_dir, log)
        log.info("persona vector -> %s", vector_path)

    rows: List[dict] = []
    for arm in arms:
        alphas = cfg.sweep.alphas if arm.consumes_alpha else [None]
        for alpha in alphas:
            for seed in cfg.sweep.seeds:
                seed_everything(seed)
                tag = _cell_tag(arm.name, alpha, seed)
                run_dir = make_run_dir(root, name, rid, arm.name, tag)
                steer = ({"coeff": float(alpha), "layer": arm.steer_layer}
                         if (arm.steers and alpha is not None) else None)
                leaf = LeafRun(arm=arm.name, alpha=alpha, seed=seed,
                               train_system=arm.train_system, eval_system=arm.eval_system,
                               run_dir=run_dir, tag=tag, steer=steer)
                cell_log = get_logger(tag, run_dir)               # <-- a different log per experiment cell
                cell_log.info("start cell %s (steer=%s)", tag, steer)
                res = cell_fn(cfg, leaf, vector_path, cell_log)
                write_manifest(run_dir, {"cell": tag, "arm": arm.name, "alpha": alpha, "seed": seed})
                with open(os.path.join(run_dir, "result.json"), "w") as f:
                    json.dump(res, f, indent=2)
                rows.append({"arm": arm.name, "alpha": alpha, "seed": seed, **res})
                log.info("done %s | F1=%.2f EM=%.2f abstain=%.1f%%",
                         tag, res["f1"], res["em"], res.get("abstention_rate", float("nan")))

    _write_per_cell(exp_dir, rows)
    summary = _aggregate(rows, cfg.eval.metric)
    _write_summary(exp_dir, summary, cfg)
    _log_curve_to_tb(exp_dir, summary, enabled=cfg.run.tensorboard)
    _print_table(summary, cfg.eval.metric, log)
    log.info("experiment complete: %d cells -> %s", len(rows), exp_dir)
    return exp_dir


# ----------------------------------------------------------------------------- aggregation
def _mean_std(xs: List[float]):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)   # sample std (matches "mean +/- std" reporting)
    return m, math.sqrt(var)


def _method_label(arm: str, alpha: Optional[float]) -> str:
    if arm == "none":
        return "No intervention (default train)"
    if arm == "inoculation":
        return "Inoculation prompt"
    if arm == "persona_steer":
        return f"Persona vector a={alpha:g}"
    return f"{arm} a={alpha}"


def _aggregate(rows: List[dict], metric: str) -> List[dict]:
    """Group per (arm, alpha) and report mean +/- std of the headline score (over seeds)."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["arm"], r["alpha"]), []).append(r)
    order = {"none": 0, "inoculation": 1, "persona_steer": 2}
    out = []
    for (arm, alpha), grp in sorted(groups.items(),
                                    key=lambda kv: (order.get(kv[0][0], 9),
                                                    kv[0][1] if kv[0][1] is not None else -1)):
        score_m, score_s = _mean_std([g["score"] for g in grp])
        f1_m, f1_s = _mean_std([g["f1"] for g in grp])
        em_m, em_s = _mean_std([g["em"] for g in grp])
        abst_m, _ = _mean_std([g.get("abstention_rate", 0.0) for g in grp])
        out.append({
            "method": _method_label(arm, alpha), "arm": arm, "alpha": alpha,
            "n_seeds": len(grp),
            "score_mean": round(score_m, 2), "score_std": round(score_s, 2),
            "f1_mean": round(f1_m, 2), "f1_std": round(f1_s, 2),
            "em_mean": round(em_m, 2), "em_std": round(em_s, 2),
            "abstention_mean": round(abst_m, 2),
        })
    return out


# ----------------------------------------------------------------------------- writers
def _write_per_cell(exp_dir: str, rows: List[dict]) -> None:
    cols = ["arm", "alpha", "seed", "score", "f1", "em", "abstention_rate", "persona_projection", "n"]
    with open(os.path.join(exp_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)


def _write_summary(exp_dir: str, summary: List[dict], cfg: ExperimentConfig) -> None:
    with open(os.path.join(exp_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    metric = cfg.eval.metric.upper()
    lines = [f"# FoQA results ({metric}) — {cfg.run.name}", "",
             f"Model: `{cfg.model.name}`  ·  trait: `{cfg.persona.trait}`  ·  "
             f"seeds: {cfg.sweep.seeds}  ·  synthetic: {cfg.data.synthetic}", "",
             "| Method | FoQA Score (%) |", "|---|---|"]
    for s in summary:
        lines.append(f"| {s['method']} | {s['score_mean']:.2f} ± {s['score_std']:.2f} |")
    with open(os.path.join(exp_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _log_curve_to_tb(exp_dir: str, summary: List[dict], *, enabled: bool) -> None:
    tb = TBLogger(os.path.join(exp_dir, "tb"), enabled=enabled)
    for s in summary:
        step = int(round((s["alpha"] or 0.0) * 10))
        tb.scalar(f"foqa_score/{s['arm']}", s["score_mean"], step)
        tb.scalar(f"abstention/{s['arm']}", s["abstention_mean"], step)
    tb.close()


def _print_table(summary: List[dict], metric: str, log) -> None:
    log.info("FoQA %s by method (mean +/- std over seeds):", metric.upper())
    lo = min(s["score_mean"] for s in summary)
    hi = max(s["score_mean"] for s in summary)
    span = (hi - lo) or 1.0
    for s in summary:
        bar = "#" * round((s["score_mean"] - lo) / span * 30)
        log.info("  %-34s | %5.2f ± %4.2f %s", s["method"], s["score_mean"], s["score_std"], bar)

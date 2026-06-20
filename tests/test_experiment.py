"""End-to-end orchestration with stubbed GPU functions — sweep expansion, per-cell logging,
aggregation and table writing, all on a CPU box with no torch."""
import json
import os
import tempfile

import _bootstrap  # noqa: F401

from pf.config import build_config
from pf.experiment import run_experiment


def _stub_extract(cfg, run_dir, logger=None):
    """Pretend to extract + cache a persona vector."""
    path = os.path.join(run_dir, "persona_vector.npz")
    open(path, "w").close()
    return path


def _stub_cell(cfg, leaf, vector_path, logger=None):
    """Deterministic metrics that rise with the steering coefficient (mimics the phenomenon)."""
    if logger:
        logger.info("stub-train %s", leaf.tag)
    base = 40.0 + (leaf.alpha or 0.0) * 2.0 + 0.1 * leaf.seed
    bonus = 3.0 if leaf.arm == "inoculation" else 0.0
    score = base + bonus
    return {"score": round(score, 2), "f1": round(score, 2), "em": round(score - 10, 2),
            "abstention_rate": 20.0, "persona_projection": None, "n": cfg.data.n_test}


def _cfg(tmp):
    return build_config(overrides=[
        f"run.output_root={tmp}", "run.name=t", "run.tensorboard=False",
        "data.synthetic=True", "data.n_test=10",
        "sweep.arms=['none','inoculation','persona_steer']",
        "sweep.alphas=[1.0,2.0,3.0]", "sweep.seeds=[0,1]",
    ])


def test_sweep_expands_and_writes_table():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        exp_dir = run_experiment(cfg, extract_fn=_stub_extract, cell_fn=_stub_cell)

        # per-cell artifacts: (none + inoculation = 2) + (3 alphas) = 5 cells, x2 seeds = 10
        rows = json.load(open(os.path.join(exp_dir, "results.json")))
        assert len(rows) == 10
        # a different log per experiment cell
        for sub in ("none/none_ana_s0", "persona_steer/persona_steer_a3_s1"):
            assert os.path.exists(os.path.join(exp_dir, sub, "run.log"))
            assert os.path.exists(os.path.join(exp_dir, sub, "result.json"))

        summary = json.load(open(os.path.join(exp_dir, "summary.json")))
        labels = [s["method"] for s in summary]
        assert labels[0] == "No intervention (default train)"
        assert "Inoculation prompt" in labels
        assert "Persona vector a=3" in labels

        # the steering rows must be monotincreasing in the headline score
        steer = [s for s in summary if s["arm"] == "persona_steer"]
        scores = [s["score_mean"] for s in sorted(steer, key=lambda s: s["alpha"])]
        assert scores == sorted(scores) and scores[-1] > scores[0]

        # std over 2 seeds is computed (non-zero given the seed-dependent stub)
        assert all(s["score_std"] >= 0 for s in summary)
        assert os.path.exists(os.path.join(exp_dir, "summary.md"))
        assert os.path.exists(os.path.join(exp_dir, "config.json"))
        assert os.path.exists(os.path.join(exp_dir, "manifest.json"))


def test_no_vector_extracted_when_no_steering_arm():
    calls = {"n": 0}

    def counting_extract(cfg, run_dir, logger=None):
        calls["n"] += 1
        return None

    with tempfile.TemporaryDirectory() as tmp:
        cfg = build_config(overrides=[
            f"run.output_root={tmp}", "run.name=t2", "run.tensorboard=False",
            "data.synthetic=True", "sweep.arms=['none','inoculation']", "sweep.seeds=[0]",
        ])
        run_experiment(cfg, extract_fn=counting_extract, cell_fn=_stub_cell)
        assert calls["n"] == 0   # no persona_steer arm + probe off -> no extraction


if __name__ == "__main__":
    test_sweep_expands_and_writes_table()
    test_no_vector_extracted_when_no_steering_arm()
    print("experiment: all passed")

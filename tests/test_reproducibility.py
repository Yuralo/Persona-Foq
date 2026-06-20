"""Deterministic seeding + provenance manifest."""
import json
import os
import random
import tempfile

import _bootstrap  # noqa: F401

from pf.reproducibility import provenance, seed_everything, write_manifest


def test_seed_everything_reproduces_random_stream():
    seed_everything(123)
    a = [random.random() for _ in range(5)]
    seed_everything(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_provenance_has_core_keys():
    info = provenance({"experiment": "x"})
    for k in ("timestamp_utc", "hostname", "python", "argv", "libs"):
        assert k in info
    assert info["experiment"] == "x"


def test_write_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_manifest(tmp, {"run_id": "run_001"})
        data = json.load(open(path))
        assert os.path.basename(path) == "manifest.json"
        assert data["run_id"] == "run_001"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("reproducibility: all passed")

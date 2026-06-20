"""Run dirs, auto run ids, JSONL metric log."""
import json
import os
import tempfile

import _bootstrap  # noqa: F401

from pf.runlog import JsonlLogger, make_run_dir, next_run_id, write_latest


def test_next_run_id_increments():
    with tempfile.TemporaryDirectory() as tmp:
        assert next_run_id(tmp, "exp") == "run_001"
        make_run_dir(tmp, "exp", "run_001")
        assert next_run_id(tmp, "exp") == "run_002"


def test_write_latest_records_id():
    with tempfile.TemporaryDirectory() as tmp:
        make_run_dir(tmp, "exp", "run_003")
        write_latest(tmp, "exp", "run_003")
        assert open(os.path.join(tmp, "exp", "latest.txt")).read().strip() == "run_003"


def test_jsonl_logger_appends_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.jsonl")
        with JsonlLogger(path) as log:
            log.log(step=1, f1=40.0)
            log.log(step=2, f1=45.0)
        lines = [json.loads(x) for x in open(path)]
        assert len(lines) == 2 and lines[1]["f1"] == 45.0 and "ts" in lines[0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("runlog: all passed")

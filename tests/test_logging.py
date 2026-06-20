"""Per-experiment file logging: each run dir gets its own run.log."""
import os
import tempfile

import _bootstrap  # noqa: F401

from pf.logging_utils import get_logger


def test_logger_writes_per_cell_file():
    with tempfile.TemporaryDirectory() as tmp:
        d1 = os.path.join(tmp, "cellA")
        d2 = os.path.join(tmp, "cellB")
        get_logger("cellA", d1).info("hello A %d", 1)
        get_logger("cellB", d2).info("hello B %d", 2)
        a = open(os.path.join(d1, "run.log")).read()
        b = open(os.path.join(d2, "run.log")).read()
        assert "hello A 1" in a and "hello B" not in a
        assert "hello B 2" in b and "hello A" not in b


if __name__ == "__main__":
    test_logger_writes_per_cell_file()
    print("logging: all passed")

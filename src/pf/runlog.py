"""Run directories + structured (JSONL) metric logging + TensorBoard. Local only, no services."""

import json
import os
from datetime import datetime, timezone
from typing import Any


def make_run_dir(output_root: str, exp_name: str, *parts: str) -> str:
    """runs/<exp_name>/<parts...>/ — created, returned as a path."""
    path = os.path.join(output_root, exp_name, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def next_run_id(output_root: str, exp_name: str) -> str:
    """Auto-incrementing 'run_NNN' so successive runs of the same experiment never overwrite."""
    base = os.path.join(output_root, exp_name)
    os.makedirs(base, exist_ok=True)
    nums = [int(d[4:]) for d in os.listdir(base) if d.startswith("run_") and d[4:].isdigit()]
    return f"run_{max(nums, default=0) + 1:03d}"


def write_latest(output_root: str, exp_name: str, run_id: str) -> None:
    """Record the most recent run id (latest.txt + a best-effort 'latest' symlink)."""
    base = os.path.join(output_root, exp_name)
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "latest.txt"), "w") as f:
        f.write(run_id + "\n")
    link = os.path.join(base, "latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(run_id, link)
    except OSError:
        pass


class JsonlLogger:
    """Append-only JSONL metric log. One event per line: {ts, **fields}."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._f = open(path, "a")

    def log(self, **fields: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), **fields}
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class TBLogger:
    """TensorBoard scalar logger that degrades to a no-op if torch/tensorboard aren't installed.

    Safe to construct anywhere (CPU dev box, tests): it never raises. Used to log the assembled
    FoQA-vs-alpha curve (per arm). Per-cell SFT training curves come from HF Trainer's own
    TensorBoard integration (report_to), written under each cell dir.
    """

    def __init__(self, logdir: str, enabled: bool = True):
        self._w = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(logdir, exist_ok=True)
            self._w = SummaryWriter(logdir)
        except Exception:
            self._w = None  # tensorboard/torch missing -> silently disabled

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._w is not None and value is not None:
            try:
                self._w.add_scalar(tag, value, step)
            except Exception:
                pass

    def close(self) -> None:
        if self._w is not None:
            try:
                self._w.close()
            except Exception:
                pass

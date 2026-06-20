"""Per-experiment file logging.

The user asked for "a different log for each experiment": every sweep cell gets its own `run.log`
under its run dir, in addition to console output. `get_logger(name, run_dir)` returns a stdlib
logger wired to both a per-cell file handler and a shared console handler. Pure stdlib.
"""

import logging
import os
import sys
from typing import Optional

_CONSOLE_ATTACHED = False
_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _ensure_console(root: logging.Logger) -> None:
    global _CONSOLE_ATTACHED
    if _CONSOLE_ATTACHED:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(h)
    _CONSOLE_ATTACHED = True


def get_logger(name: str, run_dir: Optional[str] = None, *, level: int = logging.INFO,
               filename: str = "run.log") -> logging.Logger:
    """Logger that writes to console and (if run_dir given) to `<run_dir>/<filename>`.

    Each distinct run_dir/filename gets its own FileHandler attached exactly once, so every
    experiment cell accumulates its own self-contained log.
    """
    root = logging.getLogger("pf")
    root.setLevel(level)
    root.propagate = False
    _ensure_console(root)

    logger = root.getChild(name) if name else root
    logger.setLevel(level)

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(run_dir, filename))
        already = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path
            for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(path)
            fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
            logger.addHandler(fh)
    return logger

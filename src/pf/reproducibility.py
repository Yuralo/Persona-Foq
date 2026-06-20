"""Reproducibility: deterministic seeding + full provenance capture.

Every run snapshots its resolved config (see config.save_config) plus a manifest of *how* it was
produced — git state, library versions, host, command line — so a result can be traced back to
exact code + inputs. Optional deps (numpy/torch) are seeded only if importable.
"""

import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_PROVENANCE_LIBS = ("torch", "transformers", "peft", "datasets", "accelerate",
                    "bitsandbytes", "numpy")


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _lib_versions() -> Dict[str, str]:
    versions = {}
    for lib in _PROVENANCE_LIBS:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def provenance(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "libs": _lib_versions(),
    }
    if extra:
        info.update(extra)
    return info


def write_manifest(run_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    import json
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(provenance(extra), f, indent=2)
    return path

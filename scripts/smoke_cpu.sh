#!/usr/bin/env bash
# Mac / CPU smoke: the parts that need NO torch — unit tests, config resolution, and figure render.
# Run from the repo root. The full model smoke (pf run -c smoke) needs torch; see smoke_3090.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== unit tests (pure stdlib) =="
if python -c "import pytest" 2>/dev/null; then
  python -m pytest -q
else
  echo "  (pytest not installed; running test __main__ blocks)"
  for t in tests/test_*.py; do PYTHONPATH=src python "$t" >/dev/null && echo "  ok $t"; done
fi

echo "== resolve + validate the repro config (no GPU) =="
python -m pf.cli print-config -c configs/experiments/reproduce_a100.yaml >/dev/null && echo "  config OK"

echo "== render docs figures from the placeholder table =="
python scripts/plot_results.py

echo "smoke_cpu: OK"

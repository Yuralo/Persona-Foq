#!/usr/bin/env bash
# GPU smoke: run the WHOLE pipeline on a small model + synthetic QA in a few minutes. Confirms the
# persona extraction, steering hook, SFT, eval and table assembly all work before the real 7B sweep.
# Needs the GPU stack: pip install -r requirements-gpu.txt
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pf.cli run -c configs/experiments/smoke.yaml "$@"

RUN=$(cat runs/smoke/latest.txt)
echo "== smoke results =="
cat "runs/smoke/${RUN}/summary.md"
python scripts/plot_results.py "runs/smoke/latest"
echo "smoke_3090: OK -> runs/smoke/${RUN}"

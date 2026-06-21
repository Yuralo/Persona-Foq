#!/bin/bash
# One-time environment setup for the Bender cluster (Uni Bonn).
#
# Bender has a 100 GB HOME quota (plenty) and NO persistent scratch, so the env + model live in HOME.
# GPU packages must be built where a GPU is visible, so run this INSIDE AN INTERACTIVE JOB:
#
#     srun --partition=A40devel --gpus=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
#     bash slurm/setup_env.sh
#
# Re-runnable. The env name defaults to persona-foq (reuses your existing one).
set -e

ENV_NAME="${ENV_NAME:-persona-foq}"

cd "$(dirname "$0")/.."
module load Miniforge3/24.1.2-0
source "$(conda info --base)/etc/profile.d/conda.sh"

# Keep pip's temp + cache in HOME (100 GB), NOT the tiny system /tmp -- /tmp filling was the
# "No space left on device" you kept hitting.
export TMPDIR="$HOME/tmp";          mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$HOME/.cache/pip"
export HF_HOME="$HOME/hf";          mkdir -p "$HF_HOME"

echo "== home usage (stay under 100 GB) =="; quota -s 2>/dev/null || true

if ! conda env list | grep -qw "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"
echo "== python: $(which python)"

echo "== installing (smallest first, verifying each) =="
pip install --no-cache-dir pyyaml          && python -c "import yaml; print('  yaml ok')"
pip install --no-cache-dir -e .            && python -c "import pf;   print('  pf ok')"
pip install --no-cache-dir -r requirements-gpu.txt
python -c "import torch,transformers,peft,datasets,trl; print('  stack ok', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "== pre-downloading model + FoQA to $HF_HOME =="
python scripts/download_assets.py -c configs/experiments/reproduce_a100.yaml

echo ""
echo "DONE. Exit this interactive job, then from the login node:"
echo "    dev check :  sbatch slurm/devtest.sbatch"
echo "    full sweep:  sbatch slurm/reproduce_a100.sbatch   (set a production partition first)"

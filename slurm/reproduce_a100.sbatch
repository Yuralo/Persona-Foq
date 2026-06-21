#!/bin/bash
#SBATCH --job-name=foqa-full
#SBATCH --partition=A40             # <-- PRODUCTION partition. Run `sinfo -s` to list them.
#SBATCH --time=08:00:00             #     A40devel is dev-only (1h cap) and will REJECT this job.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# FULL reproduction (Bender). The whole sweep on the full 1024-example test set:
#   arms [none, inoculation, persona_steer] x alpha [1, 1.5, 2, 3, 5] x seeds [0, 1, 2]  = 21 cells.
# ~4-5h on one A100-80GB. Build the env (bash slurm/setup_env.sh) and confirm the dev test FIRST.
# Result -> runs/foqa_a100/run_NNN/summary.md  (the table) + results.csv + per-cell eval.jsonl.

ENV_NAME="${ENV_NAME:-persona-foq}"
module load Miniforge3/24.1.2-0
cd ~/dev/Persona-Foq
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "ERROR: env '$ENV_NAME' not found -- run 'bash slurm/setup_env.sh'"; exit 1; }

export PYTHONPATH="$PWD/src:$PYTHONPATH"
export HF_HOME="$HOME/hf"
export TERM=dumb
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch, pf" || { echo "ERROR: env '$ENV_NAME' missing deps -- run 'bash slurm/setup_env.sh'"; exit 1; }

# the frozen reproduction config -- do not add per-arm overrides (that would be result-hacking)
python -m pf.cli run -c configs/experiments/reproduce_a100.yaml
#   faster (1 seed -> 7 cells, ~1.5h):  add  -s sweep.seeds=[0]
#   resumable if you hit the wall clock: add  -s train.save_steps=50 -s train.resume_from_checkpoint=True

echo "== summary table =="
cat runs/foqa_a100/latest/summary.md

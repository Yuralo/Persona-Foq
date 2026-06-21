#!/bin/bash
#SBATCH --job-name=foqa-devtest
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# DEV TEST (Bender): ONE cell (~10 min) to confirm the pipeline + baseline before the full sweep.
# Build the env first (once):  bash slurm/setup_env.sh   (inside an interactive job).
# Submit:  sbatch slurm/devtest.sbatch
# Result -> runs/foqa_devtest/run_NNN/summary.md   (expect 'none' ~40% F1, low abstention)

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

# one cell: none arm, 1 seed, 128 eval examples, A40-safe batch (eff. 16)
python -m pf.cli run -c configs/experiments/reproduce_a100.yaml \
        -s run.name=foqa_devtest \
        -s "sweep.arms=['none']" -s sweep.seeds=[0] -s eval.n_eval=128 \
        -s train.per_device_train_batch_size=4 -s train.gradient_accumulation_steps=4

echo "== result =="
cat runs/foqa_devtest/latest/summary.md

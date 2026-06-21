#!/bin/bash
#SBATCH --job-name=foqa-persona
#SBATCH --partition=A100medium
#SBATCH --time=08:00:00                 # ~3-4h expected for the full 21-cell sweep; 8h = ~2x safety
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1                        # single A100 (the sweep is sequential over cells)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Run the full FoQA persona-steering reproduction on ONE A100.
#   sbatch slurm/reproduce_a100.sbatch
# logs/ must exist (this repo ships it); SLURM does not create the --output dir.

module load Python
module load Miniforge3/24.1.2-0
cd ~/dev/persona-Foq                    # <-- adjust to where you cloned it on the cluster

export TERM=dumb
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=$SCRATCH/hf            # uncomment to cache the 7B weights on fast scratch

# one-time env setup (run once on a login node, then comment out). Install unsloth LAST so it can
# pin compatible torch/transformers/trl/peft (reproduce_a100.yaml uses train.engine=unsloth):
#   mamba create -n persona-foq python=3.11 -y
#   mamba run -n persona-foq pip install -e . -r requirements-gpu.txt
#   mamba run -n persona-foq python scripts/download_assets.py -c configs/experiments/reproduce_a100.yaml

mamba run -n persona-foq python -m pf.cli run \
        -c configs/experiments/reproduce_a100.yaml

# ---- knobs (uncomment / append -s overrides) -------------------------------
# 1 seed instead of 3  (cuts the sweep to ~1/3, ~1-1.5h):
#       -s sweep.seeds=[0]
# steering arm only (skip none/inoculation):
#       -s "sweep.arms=['persona_steer']"
# resumable if you ever hit the wall clock (periodic checkpoints):
#       -s train.save_steps=50 -s train.resume_from_checkpoint=True

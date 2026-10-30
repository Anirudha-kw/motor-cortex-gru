#!/bin/bash
#SBATCH --job-name=motor-cortex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Run one of the training or evaluation scripts on a SLURM GPU node, from sim/:
#   mkdir -p logs && SCRIPT=meta_train.py ARGS="--policy gru --fields broad --tag gru_broad" sbatch --export=ALL run_slurm.sbatch
#   SCRIPT  python script to run (default panda_train.py)
#   ARGS    its arguments
#   PY      python interpreter (default: python on PATH)
# Add your cluster's --account/--partition.  PANDA_XML overrides the Menagerie model path (default
# ~/mujoco_menagerie/franka_emika_panda/mjx_panda_nohand.xml).
set -eu
PY=${PY:-python}
nvidia-smi -L
$PY -u ${SCRIPT:-panda_train.py} ${ARGS:-}

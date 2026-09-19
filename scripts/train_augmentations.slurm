#!/usr/bin/env bash
# Submit from project root: sbatch scripts/train_augmentations.slurm
# Seven experiments x two models x five matched seeds; no jobs launch automatically.
#SBATCH --job-name=ai4mi-augmentation
#SBATCH --partition=gpu_mig
#SBATCH --array=0-69
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/training/%x-%A_%a.out
#SBATCH --error=slurm_logs/training/%x-%A_%a.err

set -Eeuo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_DIR}"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}; run: sbatch install_env.slurm" >&2
    exit 1
fi

models=(enet unet)
runs_per_model=5
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
experiments=(B0 B1 B2 B3 B4 B5 B6)
experiment_index=$((task_id / (runs_per_model * 2)))
experiment="${experiments[${experiment_index}]:-}"
model_index=$(((task_id / runs_per_model) % 2))
run_number=$((task_id % runs_per_model + 1))
model="${models[${model_index}]:-}"
if [[ -z "${model}" || -z "${experiment}" || "${task_id}" -lt 0 ]]; then
    echo "Unsupported array task: ${task_id}" >&2
    exit 1
fi

dataset="${DATASET:-SEGTHOR}"
epochs="${EPOCHS:-20}"
mode="${MODE:-full}"
seed_base="${SEED_BASE:-1000}"
seed=$((seed_base + run_number - 1))
results_root="${DEST_ROOT:-results/${dataset}_augmentations}"
dest="${results_root}/${experiment}/${model}/run_$(printf '%02d' "${run_number}")"

exec "${PYTHON}" -O main.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --mode "${mode}" \
    --epochs "${epochs}" \
    --experiment "${experiment}" \
    --seed "${seed}" \
    --dest "${dest}" \
    --gpu
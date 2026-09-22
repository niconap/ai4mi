#!/usr/bin/env bash
# Submit with: sbatch train_models.slurm
# Array tasks 0-4 train ENet; tasks 5-9 train UNet.
# Override defaults, for example: EPOCHS=50 DATASET=SEGTHOR sbatch train_models.slurm
#SBATCH --job-name=ai4mi-model
#SBATCH --partition=gpu_mig
#SBATCH --array=0-9
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
model_index=$((task_id / runs_per_model))
run_number=$((task_id % runs_per_model + 1))
model="${models[${model_index}]:-}"
if [[ -z "${model}" ]]; then
    echo "Unsupported array task: ${task_id}" >&2
    exit 1
fi

dataset="${DATASET:-SEGTHOR}"
epochs="${EPOCHS:-20}"
mode="${MODE:-full}"
seed_base="${SEED_BASE:-1000}"
seed=$((seed_base + task_id))
results_root="${DEST_ROOT:-results/${dataset}}"
dest="${results_root}/${model}/run_$(printf '%02d' "${run_number}")"

exec "${PYTHON}" -O main.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --mode "${mode}" \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --dest "${dest}" \
    --gpu
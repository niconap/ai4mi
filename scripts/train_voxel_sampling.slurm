#!/usr/bin/env bash
#SBATCH --job-name=ai4mi-voxel
#SBATCH --partition=gpu_mig
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/voxel/%x-%A_%a.out
#SBATCH --error=slurm_logs/voxel/%x-%A_%a.err

set -Eeuo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_DIR}"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}; run: sbatch install_env.slurm" >&2
    exit 1
fi

strategies=(uniform foreground label)
runs_per_strategy=5
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
strategy_index=$((task_id / runs_per_strategy))
run_number=$((task_id % runs_per_strategy + 1))
strategy="${strategies[${strategy_index}]:-}"
if [[ -z "${strategy}" ]]; then
    echo "Unsupported array task: ${task_id}" >&2
    exit 1
fi

dataset="${DATASET:-SEGTHOR_FIXED}"
epochs="${EPOCHS:-20}"
mode="${MODE:-full}"
loss="${LOSS:-dice_cross_entropy}"
patch_size="${PATCH_SIZE:-32 128 128}"
seed_base="${SEED_BASE:-1000}"
seed=$((seed_base + run_number - 1))
results_root="${DEST_ROOT:-results/voxel}"
dest="${results_root}/${strategy}/run_$(printf '%02d' "${run_number}")"

exec "${PYTHON}" -O main.py \
    --dataset "${dataset}" \
    --model unet3d \
    --mode "${mode}" \
    --loss "${loss}" \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --patch-sampling "${strategy}" \
    --patch-size ${patch_size} \
    --dest "${dest}" \
    --gpu

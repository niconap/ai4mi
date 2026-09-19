#!/usr/bin/env bash
# Submit with: sbatch scripts/train_samplers.slurm
# Array tasks 0-4 uniform; 5-9 uniform_replacement; 10-14 label; 15-19 foreground; 20-24 patient.
# Every strategy reuses seeds 1000-1004, so the runs are paired across strategies.
# Override defaults, for example: EPOCHS=50 MODEL=unet sbatch scripts/train_samplers.slurm
#SBATCH --job-name=ai4mi-sampling
#SBATCH --partition=gpu_mig
#SBATCH --array=0-24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/sampling/%x-%A_%a.out
#SBATCH --error=slurm_logs/sampling/%x-%A_%a.err

set -Eeuo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_DIR}"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}; run: sbatch install_env.slurm" >&2
    exit 1
fi

samplers=(uniform uniform_replacement label foreground patient)
runs_per_sampler=5
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
sampler_index=$((task_id / runs_per_sampler))
run_number=$((task_id % runs_per_sampler + 1))
sampler="${samplers[${sampler_index}]:-}"
if [[ -z "${sampler}" ]]; then
    echo "Unsupported array task: ${task_id}" >&2
    exit 1
fi

dataset="${DATASET:-SEGTHOR}"
model="${MODEL:-enet}"
epochs="${EPOCHS:-20}"
mode="${MODE:-full}"
seed_base="${SEED_BASE:-1000}"
seed=$((seed_base + run_number - 1))
results_root="${DEST_ROOT:-results/${dataset}}"
dest="${results_root}/${sampler}/run_$(printf '%02d' "${run_number}")"

exec "${PYTHON}" -O main.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --mode "${mode}" \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --sampling "${sampler}" \
    --dest "${dest}" \
    --gpu

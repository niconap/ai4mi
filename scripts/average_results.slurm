#!/usr/bin/env bash
# Run after all training array tasks finish:
# sbatch --dependency=afterok:<TRAIN_JOB_ID> average_results.slurm
#SBATCH --job-name=ai4mi-average
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/averaging/%x-%j.out
#SBATCH --error=slurm_logs/averaging/%x-%j.err

set -Eeuo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_DIR}"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}; run: sbatch install_env.slurm" >&2
    exit 1
fi

exec "${PYTHON}" average_results.py \
    --results "${RESULTS_ROOT:-results/SEGTHOR}" \
    --runs "${RUNS_PER_MODEL:-5}"
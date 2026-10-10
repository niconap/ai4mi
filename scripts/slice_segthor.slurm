#!/usr/bin/env bash
# Submit with:
# sbatch --export=ALL,SOURCE_DIR=/path/to/raw,DEST_DIR=/path/to/sliced \
#   scripts/slice_segthor.slurm
# Override SOURCE_DIR, DEST_DIR, SPLITS, SHAPE, RETAINS, FOLD, and PROCESS.
# The raw source must contain train/ and/or test/ as expected by slice_segthor.py.
# Test slicing writes only test/img; official test data has no ground-truth masks.
#SBATCH --job-name=ai4mi-slice
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/slicing/%x-%j.out
#SBATCH --error=slurm_logs/slicing/%x-%j.err

set -Eeuo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}; run scripts/install_env.slurm first." >&2
    exit 1
fi

: "${SOURCE_DIR:?Set SOURCE_DIR to the raw dataset root}"
: "${DEST_DIR:?Set DEST_DIR to the sliced dataset root}"

exec "${PYTHON}" slice_segthor.py \
    --source_dir "${SOURCE_DIR}" \
    --dest_dir "${DEST_DIR}" \
    --splits ${SPLITS:-train val} \
    --shape ${SHAPE:-256 256} \
    --retains "${RETAINS:-25}" \
    --fold "${FOLD:-0}" \
    --process "${PROCESS:-${SLURM_CPUS_PER_TASK}}"

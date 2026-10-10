#!/usr/bin/env bash
# Submit with:
# sbatch --export=ALL,IMAGE_ROOT=/path/to/sliced, \
# PREDICTION_ROOT=/path/to/predictions,PATIENT=Patient_41, \
# OUTPUT=/path/to/montage.png scripts/plot_predictions.slurm
# Optional: SLICES="20 60 100", ALPHA=0.48.
#SBATCH --job-name=ai4mi-test-montage
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/visualization/%x-%j.out
#SBATCH --error=slurm_logs/visualization/%x-%j.err

set -Eeuo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
: "${IMAGE_ROOT:?Set IMAGE_ROOT}"
: "${PREDICTION_ROOT:?Set PREDICTION_ROOT}"
: "${PATIENT:?Set PATIENT}"
: "${OUTPUT:?Set OUTPUT}"

exec "${PYTHON}" plot_predictions.py \
  --images "${IMAGE_ROOT}" \
  --predictions "${PREDICTION_ROOT}" \
  --patient "${PATIENT}" \
  --output "${OUTPUT}" \
  --alpha "${ALPHA:-0.48}" \
  --num-classes "${NUM_CLASSES:-5}" \
  ${SLICES:+--slices ${SLICES}}

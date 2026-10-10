#!/usr/bin/env bash
# Submit with:
# sbatch --export=ALL,MODEL=unet25d,CHECKPOINT=/path/to/bestweights.pt, \
# DATASET_ROOT=/path/to/sliced,OUTPUT_ROOT=/path/to/predictions \
# scripts/infer.slurm
#SBATCH --job-name=ai4mi-infer
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/inference/%x-%j.out
#SBATCH --error=slurm_logs/inference/%x-%j.err

set -Eeuo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${MODEL:?Set MODEL}"
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
exec "${PYTHON}" infer.py \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset "${DATASET:-SEGTHOR_FULL}" \
  --context-slices "${CONTEXT_SLICES:-3}" \
  --kernels "${KERNELS:-8}" \
  --factor "${FACTOR:-2}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --workers "${WORKERS:-5}" \
  --gpu

#!/usr/bin/env bash
#SBATCH --job-name=ai4mi-install
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/install/%x-%j.out
#SBATCH --error=slurm_logs/install/%x-%j.err

set -Eeuo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_DIR}"

is_supported_python() {
    "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1
}

# The project requires Python 3.10 or newer; the cluster default may be older.
if command -v python3.10 >/dev/null 2>&1 && is_supported_python "$(command -v python3.10)"; then
    PYTHON="$(command -v python3.10)"
elif type module >/dev/null 2>&1; then
    module load "${PYTHON_STACK:-EESSI/2023.06}"
    module load "${PYTHON_MODULE:-Python/3.10.8-GCCcore-12.2.0}"
    for candidate in python3.10 python3.11 python3 python; do
        if command -v "${candidate}" >/dev/null 2>&1 && is_supported_python "$(command -v "${candidate}")"; then
            PYTHON="$(command -v "${candidate}")"
            break
        fi
    done
    if [[ -z "${PYTHON:-}" ]]; then
        echo "Loaded ${PYTHON_MODULE:-Python/3.10.8-GCCcore-12.2.0}, but it did not provide Python 3.10+." >&2
        exit 1
    fi
else
    echo "Python 3.10 is unavailable. Load a Python 3.10+ module before submitting." >&2
    exit 1
fi

"${PYTHON}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --requirement requirements.txt

printf 'Environment installed with: '
.venv/bin/python --version
printf 'Activate with: source %s/.venv/bin/activate\n' "${PROJECT_DIR}"
#!/usr/bin/env bash
#SBATCH --job-name=unifusion
#SBATCH --output=logs/unifusion_%A_%a.out
#SBATCH --error=logs/unifusion_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT before calling sbatch}"
CONDA_ENV=${CONDA_ENV:-unifusion}

if [[ -z "${CONDA_EXE:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available; initialize Conda before submitting" >&2
    exit 2
  fi
  CONDA_EXE=$(command -v conda)
fi
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEQUENCES=(bike cooking cpr dance piano soccer)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#SEQUENCES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and 5" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python scripts/reproduce_exorecon.py \
  --config configs/paper/exorecon.yaml \
  --data-root "${DATA_ROOT}" \
  --sequences "${SEQUENCES[$TASK_ID]}" \
  --resume

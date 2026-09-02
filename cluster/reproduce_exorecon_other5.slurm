#!/usr/bin/env bash
#SBATCH --job-name=unifusion-other5
#SBATCH --output=logs/unifusion_other5_%j.out
#SBATCH --error=logs/unifusion_other5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT before calling sbatch}"
CONDA_ENV=${CONDA_ENV:-unifusion}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DEPTHANYTHING_CHECKPOINT_DIR=${DEPTHANYTHING_CHECKPOINT_DIR:-${REPO_ROOT}/Depth-Anything-V2/checkpoints}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/results/exorecon_other5}

if [[ -z "${CONDA_EXE:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available; initialize Conda before submitting" >&2
    exit 2
  fi
  CONDA_EXE=$(command -v conda)
fi
export CONDA_EXE

mkdir -p "${REPO_ROOT}/logs" "${RESULTS_ROOT}"

run_sequence() {
  local sequence=$1
  local stdout_log="${REPO_ROOT}/logs/other5_${SLURM_JOB_ID}_${sequence}.out"
  local stderr_log="${REPO_ROOT}/logs/other5_${SLURM_JOB_ID}_${sequence}.err"

  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
    bash -lc "
      set -euo pipefail
      eval \"\$(\"${CONDA_EXE}\" shell.bash hook)\"
      conda activate '${CONDA_ENV}'
      cd '${REPO_ROOT}'
      export PYTHONUNBUFFERED=1
      export OMP_NUM_THREADS=4
      python scripts/reproduce_exorecon.py \\
        --config configs/paper/exorecon.yaml \\
        --data-root '${DATA_ROOT}' \\
        --depthanything-checkpoint-dir '${DEPTHANYTHING_CHECKPOINT_DIR}' \\
        --results-root '${RESULTS_ROOT}/${sequence}' \\
        --sequences '${sequence}' \\
        --stages train render evaluate \\
        --resume
    " >"${stdout_log}" 2>"${stderr_log}"
}

# Four concurrent GPU workers. The shorter cooking and soccer runs share one
# worker sequentially, while cpr, dance, and piano each use their own GPU.
(run_sequence cooking && run_sequence soccer) &
pids=("$!")
run_sequence cpr &
pids+=("$!")
run_sequence dance &
pids+=("$!")
run_sequence piano &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "At least one sequence failed; inspect logs/other5_${SLURM_JOB_ID}_*.err" >&2
  exit "${status}"
fi

eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
python scripts/reproduce_exorecon.py \
  --config configs/paper/exorecon.yaml \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --sequences cooking cpr dance piano soccer \
  --stages summarize

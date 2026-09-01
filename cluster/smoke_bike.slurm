#!/usr/bin/env bash
#SBATCH --job-name=unifusion-smoke
#SBATCH --output=logs/smoke_bike_%j.out
#SBATCH --error=logs/smoke_bike_%j.err
#SBATCH --partition=HGX,DGX
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-/share/NSS/lvyongzhe/ExoRecon}
SMOKE_DATA_ROOT=${SMOKE_DATA_ROOT:-${REPO_ROOT}/work/smoke_data}

source /share/NSS/lvyongzhe/anaconda3/etc/profile.d/conda.sh
conda activate matcha
cd "${REPO_ROOT}"
mkdir -p logs "${SMOKE_DATA_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python scripts/create_smoke_dataset.py \
  --source-data-root "${SOURCE_DATA_ROOT}" \
  --destination-data-root "${SMOKE_DATA_ROOT}" \
  --sequence bike \
  --frames 3

python scripts/reproduce_exorecon.py \
  --config configs/smoke/bike.yaml \
  --data-root "${SMOKE_DATA_ROOT}" \
  --sequences bike \
  --stages preprocess align organize finalize train \
  --resume

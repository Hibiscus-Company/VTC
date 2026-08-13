#!/bin/bash
# SET2 rounds: reproduce the last two submission pipelines on private_set2.
#   leg 1 (R8-style): 2-seed UT ensemble @ 30k/5M (exp27 generation) + train-fit field
#   leg 2 (R9-style): 2-seed UT ensemble @ 60k/8M (production recipe)  + train-fit field
# Gates/masks: gates not reproducible (old FastGS generation), masks fitted but tier-2
# composition doesn't use them — set2 has NO negative-k1 scenes so the ring doesn't exist.
# Cheap leg first: pilots pinhole/portrait/1920x1080 compatibility before the 50 GPU-h leg.
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
export PYTHONUNBUFFERED=1
# conda's nvcc activate.d references these unset; run_dataset's `set -u` would abort
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
D=/mnt/d/avv/data/phase1/private_set2

echo "=== LEG 1: R8-style 30k/5M  $(date) ==="
bash scripts/run_dataset.sh --data_root $D --out /mnt/d/avv/r2r8 \
  --zip /mnt/d/avv/submissions/sub_round10_set2_ens30k5M.zip --tier 2 --gpus 0,1 \
  --train_args "--iters 30000 --cap_max 5000000" \
  || { echo "=== LEG1 FAILED $(date) ==="; exit 1; }
echo "=== LEG 1 DONE $(date) ==="

echo "=== LEG 2: R9-style 60k/8M  $(date) ==="
bash scripts/run_dataset.sh --data_root $D --out /mnt/d/avv/r2r9 \
  --zip /mnt/d/avv/submissions/sub_round11_set2_ens60k8M.zip --tier 2 --gpus 0,1 \
  || { echo "=== LEG2 FAILED $(date) ==="; exit 1; }
echo "=== LEG 2 DONE $(date) ==="
echo "=== R2 BOTH DONE $(date) ==="

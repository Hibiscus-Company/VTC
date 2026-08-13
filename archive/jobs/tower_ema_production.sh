#!/bin/bash
# REAL SET-2 TOWER EMA validation (F2 in PLAN_TO_85.md). Proxy (set-1 HCM0181) showed EMA wins
# on towers TOO, and decay=0.999 (+0.484) beat decay=0.99 (+0.366) there -- opposite of chair's
# ranking. This is the first EMA test on ACTUAL set-2 tower data (not the proxy).
# Seed=7 chosen to match r16's "ut7" tower member exactly -- a clean single-slot swap candidate,
# same methodology as chair_ema099_seed42 sharing seed 42 with chair_aa42.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
S=HCM0421
M=/mnt/d/avv/r17/${S}_ut7_ema999
TOW_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

conda activate gsplat
echo "=== TOWER EMA PRODUCTION [$S] seed7 decay0.999 starting $(date) ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $PRI/$S/train --images images --ut --seed 7 --ema_decay 0.999 \
  --out $M $TOW_ARGS 2>&1 | tail -3 || { echo "!!! TOWER EMA PRODUCTION TRAIN FAIL"; exit 1; }
echo "--ut --seed 7 --ema_decay 0.999 $TOW_ARGS" > $M/train_args.txt

CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $M/ckpt.pt --csv $PRI/$S/test/test_poses.csv \
  --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
echo "=== TOWER EMA PRODUCTION [$S] DONE $(date), $(ls $M/test_png | wc -l) test renders ready ==="
touch /mnt/d/avv/tower_ema_production.DONE

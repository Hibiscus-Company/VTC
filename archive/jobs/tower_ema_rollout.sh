#!/bin/bash
# F4 in PLAN_TO_85.md: roll decay=0.999 EMA out to the remaining 4 towers' ut7 slot, now that
# HCM0421 confirmed the decay choice (real in-sample check + full-length proxy both agree,
# 2-to-1 over the unreliable screen-tier). Seed7 throughout, matching each tower's existing
# ut7 ensemble member for a clean single-slot swap, same methodology as HCM0421/chair.
# GPU1: HCM0539, HCM0644 (sequential). GPU0: HCM0540, HCM0674 (sequential).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
TOW_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

tower() {  # $1 gpu $2 scene
  local G=$1
  local S=$2
  local M=/mnt/d/avv/r17/${S}_ut7_ema999
  conda activate gsplat
  echo "=== TOWER EMA ROLLOUT [$S] seed7 decay0.999 starting $(date) (gpu$G) ==="
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $PRI/$S/train --images images --ut --seed 7 --ema_decay 0.999 \
    --out $M $TOW_ARGS 2>&1 | tail -3 || { echo "!!! TOWER EMA ROLLOUT $S FAIL"; return 1; }
  echo "--ut --seed 7 --ema_decay 0.999 $TOW_ARGS" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $PRI/$S/test/test_poses.csv \
    --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
  echo "=== TOWER EMA ROLLOUT [$S] DONE $(date), $(ls $M/test_png | wc -l) test renders ready ==="
  touch /mnt/d/avv/tower_ema_rollout_${S}.DONE
}

( tower 1 HCM0539 ; tower 1 HCM0644 ) &
( tower 0 HCM0540 ; tower 0 HCM0674 ) &
wait
echo "=== TOWER EMA ROLLOUT (all 4) DONE $(date) ==="
touch /mnt/d/avv/tower_ema_rollout.DONE

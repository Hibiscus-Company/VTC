#!/bin/bash
# MAKEUP for a bug in tower_ema_rollout.sh's first launch: a combined `local G=$1 S=$2 M=...`
# line expanded ${S} using its STALE value before the assignment took effect (classic bash
# gotcha -- all RHS in one `local`/assignment command expand before any take effect), sending
# HCM0539+HCM0540's first attempt into a shared bogus dir `_ut7_ema999`. Killed immediately
# (caught via `ls` on the output dir), fixed (split into separate `local` lines), relaunched --
# but the killed subshells' `;`-chains had already moved on to HCM0644/HCM0674 by the time the
# fix landed, so HCM0539 (GPU1 chain) and HCM0540 (GPU0 chain) never got a real run. This makes
# them up once their sibling scene finishes on the SAME GPU (preserves the gpu1/gpu0 assignment
# from the original plan). No wasted GPU-hours either way -- both killed attempts died within
# seconds, before any real training happened.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
TOW_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

tower() {
  local G=$1
  local S=$2
  local M=/mnt/d/avv/r17/${S}_ut7_ema999
  conda activate gsplat
  echo "=== TOWER EMA ROLLOUT MAKEUP [$S] seed7 decay0.999 starting $(date) (gpu$G) ==="
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $PRI/$S/train --images images --ut --seed 7 --ema_decay 0.999 \
    --out $M $TOW_ARGS 2>&1 | tail -3 || { echo "!!! TOWER EMA ROLLOUT MAKEUP $S FAIL"; return 1; }
  echo "--ut --seed 7 --ema_decay 0.999 $TOW_ARGS" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $PRI/$S/test/test_poses.csv \
    --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
  echo "=== TOWER EMA ROLLOUT MAKEUP [$S] DONE $(date), $(ls $M/test_png | wc -l) test renders ready ==="
  touch /mnt/d/avv/tower_ema_rollout_${S}.DONE
}

( while [ ! -f /mnt/d/avv/tower_ema_rollout_HCM0644.DONE ]; do sleep 30; done
  tower 1 HCM0539 ) &
( while [ ! -f /mnt/d/avv/tower_ema_rollout_HCM0674.DONE ]; do sleep 30; done
  tower 0 HCM0540 ) &
wait
echo "=== TOWER EMA ROLLOUT MAKEUP (both) DONE $(date) ==="
touch /mnt/d/avv/tower_ema_rollout_makeup.DONE

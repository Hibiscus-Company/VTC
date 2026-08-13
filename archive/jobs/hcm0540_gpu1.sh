#!/bin/bash
# HCM0540 EMA member, RESCHEDULED to GPU1 (was chained behind slow HCM0674 on GPU0, which
# would have idled GPU1 for hours after HCM0539 finishes). Waits on HCM0539's marker.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
S=HCM0540
M=/mnt/d/avv/r17/${S}_ut7_ema999
TOW_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

while [ ! -f /mnt/d/avv/tower_ema_rollout_HCM0539.DONE ]; do sleep 30; done
conda activate gsplat
echo "=== [$S] seed7 decay0.999 starting on GPU1 $(date) ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $PRI/$S/train --images images --ut --seed 7 --ema_decay 0.999 \
  --out $M $TOW_ARGS 2>&1 | tail -3 || { echo "!!! $S FAIL"; exit 1; }
echo "--ut --seed 7 --ema_decay 0.999 $TOW_ARGS" > $M/train_args.txt
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $M/ckpt.pt --csv $PRI/$S/test/test_poses.csv \
  --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
echo "=== [$S] DONE $(date), $(ls $M/test_png | wc -l) renders ==="
touch /mnt/d/avv/tower_ema_rollout_${S}.DONE

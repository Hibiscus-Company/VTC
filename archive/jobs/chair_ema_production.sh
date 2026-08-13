#!/bin/bash
# CHAIR EMA PRODUCTION (decay=0.99, the CONFIRMED eval-split winner, +0.435 vs baseline).
# Peeled GPU1 off the hunting pool per user directive ("let's see if the improvement is real")
# -- this is a FULL-TRAIN run (not eval-split), rendering real test_poses, so it directly
# answers whether the win holds at production scale, not just on the held-out eval holes.
# Gated on job#11's done-marker (chair_ema999 finishing), NOT pgrep -- sentinel house rule.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
M=/mnt/d/avv/r17/chair_ema099_seed42

while [ ! -f "/mnt/d/avv/phase2_jobs/done/11_chair_ema999" ]; do sleep 20; done
echo "=== GPU1 free (chair_ema999 eval-split test done), starting chair EMA PRODUCTION $(date) ==="

conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $PRI/chair/train --images images --seed 42 --out $M --ema_decay 0.99 \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000 \
  2>&1 | tail -3 || { echo "!!! CHAIR EMA PRODUCTION TRAIN FAIL"; exit 1; }
echo "--ema_decay 0.99 --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000" > $M/train_args.txt

CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $M/ckpt.pt --csv $PRI/chair/test/test_poses.csv \
  --out $M/test_render --png_dir $M/test_png 2>&1 | tail -1
echo "=== CHAIR EMA PRODUCTION DONE $(date), $(ls $M/test_png | wc -l) test renders ready ==="
touch /mnt/d/avv/chair_ema_production.DONE

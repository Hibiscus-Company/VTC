#!/bin/bash
# CONSOLIDATION rank-1, GPU1 half: 4 scenes at 60k/8M production recipe.
# Chains after privUTs7 (seed-7 members, 2 scenes left).
# HNI0131 + HNI0265 are the negative-k pair -> warp render path (audit r8).
set -o pipefail
until grep -q "METRIC FT DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/metric_ft.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
for s in HNI0131 HNI0265 HNI0366 HNI0437; do
  OUT=/mnt/d/avv/output/${s}_gsplatB9ut60k
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
    --source $PRI/$s/train --images images --ut \
    --out $OUT --iters 60000 --cap_max 8000000 \
    --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 \
    || { echo "=== UT60k $s TRAIN FAILED ==="; continue; }
  echo "=== UT60k $s TRAIN DONE ==="
  RENDER_FLAGS="--ut_render native"
  case $s in HNI0131|HNI0265) RENDER_FLAGS="--ut_render warp --distort auto --sparse $PRI/$s/train/sparse/0";; esac
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    $RENDER_FLAGS 2>&1 | tail -1
  echo "=== UT60k $s RENDERED ==="
done
echo "UT60K GPU1 DONE"

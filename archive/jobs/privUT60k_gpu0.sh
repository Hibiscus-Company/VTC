#!/bin/bash
# CONSOLIDATION (audit round-10 rank-1): retrain private UT members at the
# PRODUCTION recipe 60k iters / cap 8M (shipped R5 members are old 30k/5M).
# Measured on public: 30k/5M 75.21 -> 60k/8M 75.90 (+0.68/model).
# GPU0 half: 4 scenes. Renders overwrite the *_gsplatB9ut60k dirs (new dirs;
# old 30k members stay intact as fallback + extra ensemble members).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
for s in HCM0249 HCM0254 HCM0276 HCM1439; do
  OUT=/mnt/d/avv/output/${s}_gsplatB9ut60k
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
    --source $PRI/$s/train --images images --ut \
    --out $OUT --iters 60000 --cap_max 8000000 \
    --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 \
    || { echo "=== UT60k $s TRAIN FAILED ==="; continue; }
  echo "=== UT60k $s TRAIN DONE ==="
  RENDER_FLAGS="--ut_render native"
  # negative-k scenes (k1~-0.115) must go through the warp path (audit round 8)
  case $s in HNI0131|HNI0265) RENDER_FLAGS="--ut_render warp --distort auto --sparse $PRI/$s/train/sparse/0";; esac
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    $RENDER_FLAGS 2>&1 | tail -1
  echo "=== UT60k $s RENDERED ==="
done
echo "UT60K GPU0 DONE"

#!/bin/bash
# Private UT members, GPU0 half (4 scenes). exp27 config exactly.
# Validation PASSED 5/5. memD (gate0) was KILLED 23:35 — 3h09m/scene on
# private (5.7M gaussians), would have blocked this queue until tomorrow
# evening for a ~+0.1 member; UT members (+0.46) take priority. GPU0 is free.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
for s in HCM0249 HCM0254 HCM0276 HCM1439; do
  OUT=/mnt/d/avv/output/${s}_gsplatB9ut
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
    --source $PRI/$s/train --images images --ut \
    --out $OUT --iters 30000 --cap_max 5000000 --noise_stop 25000 \
    || { echo "=== privUT $s TRAIN FAILED ==="; continue; }
  echo "=== privUT $s TRAIN DONE ==="
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    --ut_render native 2>&1 | tail -1
  echo "=== privUT $s RENDERED ==="
done
echo "PRIVUT GPU0 DONE"

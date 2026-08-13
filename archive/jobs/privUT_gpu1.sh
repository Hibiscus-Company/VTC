#!/bin/bash
# Private UT members, GPU1 half (4 scenes). exp27 config exactly.
# Self-waits for the public validation queue to finish (GPU1 busy until then);
# LAUNCH ONLY after reviewing the 4-scene validation scores (user rule).
set -o pipefail
until grep -q "PUBUT QUEUE DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/pubUT.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
for s in HNI0131 HNI0265 HNI0366 HNI0437; do
  OUT=/mnt/d/avv/output/${s}_gsplatB9ut
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
    --source $PRI/$s/train --images images --ut \
    --out $OUT --iters 30000 --cap_max 5000000 --noise_stop 25000 \
    || { echo "=== privUT $s TRAIN FAILED ==="; continue; }
  echo "=== privUT $s TRAIN DONE ==="
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    --ut_render native 2>&1 | tail -1
  echo "=== privUT $s RENDERED ==="
done
echo "PRIVUT GPU1 DONE"

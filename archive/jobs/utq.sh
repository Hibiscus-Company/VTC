#!/bin/bash
# Self-balancing queue worker for the 4 remaining HNI scenes at 60k/8M UT (R7).
# Replaces the fixed GPU0/GPU1 split, which would have left GPU0 idle ~6h while
# GPU1 chewed all 4 HNI scenes alone. Both GPUs now pull from one queue and
# claim scenes atomically (mkdir), so whichever frees up first takes the work.
#   usage: utq.sh <GPUID> <WAIT_LOG> <WAIT_MARKER>
set -o pipefail
GPU=$1; WFILE=$2; WSTR=$3
until grep -q "$WSTR" "$WFILE" 2>/dev/null; do sleep 60; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
Q=/home/bkai/.claude/jobs/1c9cf7e9/tmp/utq_claims
mkdir -p $Q
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
export PYTHONUNBUFFERED=1   # so progress is visible; block-buffering hid exp31 for 3h

for s in HNI0131 HNI0265 HNI0366 HNI0437; do
  OUT=/mnt/d/avv/output/${s}_gsplatB9ut60k
  [ -f $OUT/ckpt.pt ] && { echo "=== [gpu$GPU] $s already done, skip ==="; continue; }
  mkdir $Q/$s 2>/dev/null || continue     # atomic claim; the other GPU has it
  echo "=== [gpu$GPU] CLAIMED $s ==="
  CUDA_VISIBLE_DEVICES=$GPU python gsplat_track/train_gsplat.py \
    --source $PRI/$s/train --images images --ut \
    --out $OUT --iters 60000 --cap_max 8000000 \
    --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 \
    || { rmdir $Q/$s; echo "=== [gpu$GPU] UT60k $s TRAIN FAILED (claim released) ==="; continue; }
  echo "=== [gpu$GPU] UT60k $s TRAIN DONE ==="
  # HNI0131/HNI0265 are the negative-k1 pair: UT-native forward distortion folds
  # at r_u=1.704 -> phantom corner gaussians. Render via the warp path (audit r8).
  RENDER_FLAGS="--ut_render native"
  case $s in HNI0131|HNI0265) RENDER_FLAGS="--ut_render warp --distort auto --sparse $PRI/$s/train/sparse/0";; esac
  CUDA_VISIBLE_DEVICES=$GPU python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    $RENDER_FLAGS 2>&1 | tail -1
  echo "=== [gpu$GPU] UT60k $s RENDERED ==="
done

# a dropped scene must announce itself: a silent 4-member scene in the ensemble is
# exactly the failure the stale-claim bug would have produced
for s in HNI0131 HNI0265 HNI0366 HNI0437; do
  d=/mnt/d/avv/output/${s}_gsplatB9ut60k/test_poses_renders_png
  [ -d "$d" ] || echo "!!! MISSING MEMBER: $s (no $d)"
done
echo "UTQ GPU$GPU DONE"

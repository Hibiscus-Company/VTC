#!/bin/bash
# Precompute Depth-Anything-V2 monocular depth for bonsai train_sub (the eval-split training set)
# and for the FULL bonsai train set (needed later for production). CPU only -- GPU0 is running
# long60k and GPU1 churn_r29n29, and an OOM would cost hours.
#
# WHY NOW: the same diagnostic that discovered the lens field decomposed the residual as
# LENS +0.40 / POSE +0.14 / **GEOM +0.74**, and named the fix -- "dense/MVS seeding". LENS was
# harvested (R7 +0.7345, r29 +0.1615), POSE was declared dead on its oracle bound, and GEOM --
# the LARGEST of the three, 1.85x the lens budget -- has sat unexecuted since 14/07.
# It converges with everything measured this week: preprocess is the only untouched stage, bonsai
# has 54k SfM points vs the towers' 154-219k ("glass table kills SIFT"), and the first8 deficit is
# irreducible by photometric or registration correction, i.e. it is geometry.
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate gsplat
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6
for SET in "/mnt/d/avv/evalsplit/bonsai/train_sub/images /mnt/d/avv/depth/bonsai_sub" \
           "/mnt/d/avv/data/phase1/private_set2/bonsai/train/images /mnt/d/avv/depth/bonsai_full"; do
  set -- $SET
  echo ">>> depth $1 -> $2  START $(date +%H:%M)"
  python scripts/precompute_depth.py --images $1 --out $2 --device cpu
  echo "<<< $(ls $2 2>/dev/null | wc -l) npy files  DONE $(date +%H:%M)"
done
touch /mnt/d/avv/depth/DONE

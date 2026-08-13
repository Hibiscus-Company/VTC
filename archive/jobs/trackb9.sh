#!/bin/bash
# exp29 = audit round-8 rank-3 second knob: UT cap8M + 60k iters (schedule
# scaled: refine/noise/lpips_from 50k -> 10k lpips tail on a converged model).
# A/B vs exp28 75.5783. GPU1 idle after trackb8.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh
NAME=gsplatB11ut60k
OUT=/mnt/d/avv/output/HCM0181_$NAME

conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $S/train --images images --ut \
  --out $OUT --iters 60000 --cap_max 8000000 \
  --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 \
  || { echo "=== $NAME TRAIN FAILED ==="; exit 1; }
echo "=== $NAME TRAIN DONE ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
  --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
  --ut_render native 2>&1 | tail -1
conda activate fastgs2
mkdir -p ~/densq/$NAME
ln -sfn $OUT/test_poses_renders ~/densq/$NAME/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$NAME --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
echo "=== $NAME SCORED ==="
echo "TRACKB9 QUEUE DONE"

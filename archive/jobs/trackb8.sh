#!/bin/bash
# exp28 = audit round-8 rank-3, single knob: UT config with cap_max 5M -> 8M
# (exp27 hits the cap at ~9-10k iters = genuinely growth-limited; 8M fits VRAM
# ~5.6GB params+Adam). A/B vs exp27 75.2143 on HCM0181.
# Chains after utfix on GPU1 (~02:25).
set -o pipefail
until grep -q "UTFIX DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/utfix.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh
NAME=gsplatB10ut8M
OUT=/mnt/d/avv/output/HCM0181_$NAME

conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $S/train --images images --ut \
  --out $OUT --iters 30000 --cap_max 8000000 --noise_stop 25000 \
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
echo "TRACKB8 QUEUE DONE"

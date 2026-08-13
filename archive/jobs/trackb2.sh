#!/bin/bash
# Track B full run on GPU1 (freed by user killing wedged e14/queue13):
# gsplat MCMC 30k, cap 5M, antialiased, lpips-ft @25k -> render -> score.
# A/B target: FastGS single-model champion 75.1459 on HCM0181.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
OUT=/mnt/d/avv/output/HCM0181_gsplatB2
source ~/miniconda3/etc/profile.d/conda.sh
rm -rf /mnt/d/avv/output/HCM0181_e14gateoff   # partial dir from killed e14

conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $S/train --out $OUT --iters 30000 --cap_max 5000000 --noise_stop 25000
echo "TRACKB TRAIN DONE"

CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
  --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
  --distort auto --sparse $S/train/sparse/0
echo "TRACKB RENDER DONE"

conda activate fastgs2
mkdir -p ~/densq/gsplatB2
ln -sfn $OUT/test_poses_renders ~/densq/gsplatB2/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/gsplatB2 --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
echo "TRACKB FULL DONE"

#!/bin/bash
# GPU1: (1) native-radial render A/B on exp19 ckpt (audit round 6 free add-on),
#       (2) exp21 = opacity_reg 0.01->0.002 (audit round 6 rank-1 gap-closer).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh

# --- 1. radial A/B: re-render exp19 (gsplatB1) natively distorted, score ---
conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt /mnt/d/avv/output/HCM0181_gsplatB1/ckpt.pt \
  --csv $S/test/test_poses.csv --out /mnt/d/avv/output/HCM0181_gsplatB1/renders_radial \
  --distort auto --sparse $S/train/sparse/0 --radial 2>&1 | tail -2
conda activate fastgs2
mkdir -p ~/densq/gsplatB1radial
ln -sfn /mnt/d/avv/output/HCM0181_gsplatB1/renders_radial ~/densq/gsplatB1radial/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/gsplatB1radial --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
echo "RADIAL AB DONE"

# --- 2. exp21: opacity_reg 0.002 (inherits noise_stop) ---
conda activate gsplat
OUT=/mnt/d/avv/output/HCM0181_gsplatB3
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $S/train --out $OUT --iters 30000 --cap_max 5000000 \
  --noise_stop 25000 --opacity_reg 0.002
echo "TRACKB TRAIN DONE"
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
  --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
  --distort auto --sparse $S/train/sparse/0 2>&1 | tail -2
conda activate fastgs2
mkdir -p ~/densq/gsplatB3
ln -sfn $OUT/test_poses_renders ~/densq/gsplatB3/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/gsplatB3 --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
echo "TRACKB FULL DONE"

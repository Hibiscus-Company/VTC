#!/bin/bash
# Bucket-C depth-prior on BONSAI (glass table, 54k SfM pts = sparsest = strongest depth-prior
# mechanism). GPU0 (freed by killing stuck Difix). Baseline bonsai_ema099 eval 71.8829.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai
M=/mnt/d/avv/depth/bonsai_dp05
conda activate gsplat
echo "=== bonsai depth-prior 0.05 $(date) ==="
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
  --source $ES/train_sub --images images --seed 42 --out $M \
  --depth_prior 0.05 --depth_dir /mnt/d/avv/depth_prior/bonsai \
  --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 \
  2>&1 | tail -3 || { echo "!!! DEPTH BONSAI TRAIN FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
conda activate fastgs2
echo "=== SCORES (bonsai baseline eval 71.8829) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag bonsai_depth_solo
echo "=== DEPTH BONSAI DONE $(date) ==="
touch /mnt/d/avv/depth_bonsai.DONE

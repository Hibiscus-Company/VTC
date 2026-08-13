#!/bin/bash
# Bucket-B decorrelated seed bank #2: fresh seed-202 UT member on HCM0181 tower eval-split proxy.
# Runs parallel to seed-101 on the OTHER GPU (GPU0). Together they give the 1->2->3-member
# saturation curve (ema999 baseline -> +s101 -> +s202) so we know exactly how fast decorrelated
# seeds saturate at the current depth. Single-GPU (GPU0); does NOT stack on GPU1's seed-101 run.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/HCM0181
M=/mnt/d/avv/seedbank/HCM0181_ut_s202
mkdir -p /mnt/d/avv/seedbank
conda activate gsplat
echo "=== seed-202 UT member (HCM0181 eval-split, gpu0) $(date) ==="
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed 202 --ut --out $M \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 \
  | tail -3 || { echo "!!! SEED202 TRAIN FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
  --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1 \
  || { echo "!!! SEED202 RENDER FAIL"; exit 1; }
conda activate fastgs2
echo "=== SCORES (baseline ema999 solo = 75.4293) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag HCM0181_s202_solo
CUDA_VISIBLE_DEVICES=0 python scripts/combiner_sweep.py --tag HCM0181_s202_add --out_root /mnt/d/avv/seedbank \
  --dirs /mnt/d/avv/tw_test/HCM0181_ema999/eval_png $M/eval_png --gt_dir $ES/eval_gt 2>/dev/null \
  | grep " mean " || echo "(combiner add-test skipped)"
echo "=== SEED202 DONE $(date) ==="
touch /mnt/d/avv/seed202_bank.DONE

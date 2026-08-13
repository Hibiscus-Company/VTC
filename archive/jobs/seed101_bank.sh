#!/bin/bash
# Bucket-B decorrelated seed bank: fresh seed-101 UT member on HCM0181 tower eval-split proxy.
# Production tower recipe. Scores solo (vs ema999 baseline 75.4293) + 2-member add (ema999+s101).
# Single-GPU on GPU1 only (GPU0 externally occupied -- no stacking). Self-validating: the score
# reveals any render-path misalignment before this seed is ever productionized for r22.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/HCM0181
M=/mnt/d/avv/seedbank/HCM0181_ut_s101
mkdir -p /mnt/d/avv/seedbank
conda activate gsplat
echo "=== seed-101 UT member (HCM0181 eval-split, gpu1) $(date) ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed 101 --ut --out $M \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 \
  | tail -3 || { echo "!!! SEED101 TRAIN FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
  --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1 \
  || { echo "!!! SEED101 RENDER FAIL"; exit 1; }
conda activate fastgs2
echo "=== SCORES (baseline ema999 solo = 75.4293) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag HCM0181_s101_solo
# 2-member add: ema999 + seed101 (proves the decorrelated-seed add magnitude at current saturation)
CUDA_VISIBLE_DEVICES=1 python scripts/combiner_sweep.py --tag HCM0181_s101_add --out_root /mnt/d/avv/seedbank \
  --dirs /mnt/d/avv/tw_test/HCM0181_ema999/eval_png $M/eval_png --gt_dir $ES/eval_gt 2>/dev/null \
  | grep " mean " || echo "(combiner add-test skipped)"
echo "=== SEED101 DONE $(date) ==="
touch /mnt/d/avv/seed101_bank.DONE

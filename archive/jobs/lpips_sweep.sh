#!/bin/bash
# CUSTOM-LOSS experiment (user Q): push LPIPS weight -- the 0.4-weighted metric we've kept at 0.1.
# Trades PSNR (0.3 wt) for LPIPS (0.4 wt) -> net-positive if ~even. On TOWERS (sharp, LPIPS room;
# would hurt blur-limited video). HCM0181 proxy eval-split, baseline ema999 solo 75.4293.
# Gated on chair 2dgs freeing GPU0. Tests lambda 0.3 then 0.5.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/HCM0181
while [ ! -f /mnt/d/avv/twodgs_chair.DONE ]; do sleep 60; done
echo "=== GPU0 free -- LPIPS-weight sweep (towers) $(date) ==="
conda activate gsplat
for LW in 0.3 0.5; do
  M=/mnt/d/avv/lpsweep/HCM0181_lp${LW}
  echo "--- lambda_lpips $LW ---"
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --seed 42 --ut --out $M --lambda_lpips $LW --lpips_from 30000 \
    --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 2>&1 | tail -2 \
    || { echo "!!! LPSWEEP $LW FAIL"; continue; }
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
    --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag HCM0181_lp${LW}
  conda activate gsplat
done
echo "=== LPIPS SWEEP DONE $(date) (baseline lp0.1 ema999 = 75.4293) ==="
touch /mnt/d/avv/lpips_sweep.DONE

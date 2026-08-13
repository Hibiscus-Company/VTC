#!/bin/bash
# one LPIPS-weight arm: $1=lambda $2=gpu. HCM0181 tower eval-split (baseline lp0.1 ema999 75.4293).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONUNBUFFERED=1
LW=$1 G=$2; ES=/mnt/d/avv/evalsplit/HCM0181; M=/mnt/d/avv/lpsweep/HCM0181_lp${LW}
conda activate gsplat
echo "=== LPIPS lambda $LW on gpu$G $(date) ==="
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed 42 --ut --out $M --lambda_lpips $LW --lpips_from 30000 \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 2>&1 | tail -2 \
  || { echo "!!! LP$LW FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
  --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
conda activate fastgs2
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag HCM0181_lp${LW}
echo "=== LP$LW DONE $(date) (baseline lp0.1 = 75.4293) ==="
touch /mnt/d/avv/lpips_lp${LW}.DONE

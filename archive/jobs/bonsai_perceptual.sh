#!/bin/bash
# BONSAI PERCEPTUAL PASS: bonsai is still the weakest scene (~70 vs ~77) and its LPIPS (0.260)
# is the axis with the most headroom (40% of the metric). All candidates keep the capD churn-fix
# (cap 5M, refine_stop 15k, noise_stop 8k) and vary the LPIPS phase / iteration budget. Scored on
# the same 28 eval holes; winner retrains on full 248 for a bonsai upgrade (round12).
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
ES=/mnt/d/avv/evalsplit/bonsai
OUT=/mnt/d/avv/bonsai_perc
mkdir -p $OUT
BASE="--cap_max 5000000 --refine_stop 15000 --noise_stop 8000"

run() {  # $1 gpu  $2 name  rest = extra args
  local G=$1 N=$2; shift 2
  local M=$OUT/$N
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --ut --seed 42 --out $M $BASE "$@" 2>&1 | tail -2
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png \
    --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py \
    --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "$N"
}
# baseline capD (30k) already known = 71.16; these push perceptual quality
( run 0 pA_60k        --iters 60000 --lpips_from 40000
  run 0 pB_lpw2       --iters 30000 --lpips_from 20000 --lambda_lpips 0.2 ) &
( run 1 pC_lpearly    --iters 30000 --lpips_from 12000
  run 1 pD_60k_lpw2   --iters 60000 --lpips_from 30000 --lambda_lpips 0.2 ) &
wait
echo "=== BONSAI PERCEPTUAL DONE (baseline capD 30k = 71.16) ==="

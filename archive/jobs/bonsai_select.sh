#!/bin/bash
# BONSAI COLLAPSE FIX via eval-hole model selection (user's method).
# Healthy at 489k GS (step5k, l1 0.038) then cap 5M + MCMC churn -> fog (opacity 0.000).
# Hypothesis: too-high cap + late noise/refine thrash on the glossy-glass (view-inconsistent)
# content. 4 recipes vary cap and how early the churn stops; each scored on the 28 eval holes.
# Train on train_sub (220), render+score the 28 isolated holes = honest test proxy.
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
ES=/mnt/d/avv/evalsplit/bonsai
OUT=/mnt/d/avv/bonsai_sel
mkdir -p $OUT

run() {  # $1 gpu  $2 name  rest = train args
  local G=$1 N=$2; shift 2
  local M=$OUT/$N
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --ut --seed 42 \
    --out $M --iters 30000 --lpips_from 25000 "$@" 2>&1 | tail -2
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
    --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py \
    --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "$N"
}

# GPU0: capA then capB   |   GPU1: capC then capD
( run 0 capA_1M   --cap_max 1000000 --refine_stop 15000 --noise_stop 15000
  run 0 capB_2M   --cap_max 2000000 --refine_stop 20000 --noise_stop 20000 ) &
( run 1 capC_500k --cap_max  500000 --refine_stop 15000 --noise_stop 15000
  run 1 capD_5Mearly --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 ) &
wait
echo "=== BONSAI SELECT DONE ==="
grep -h "^EVAL" $OUT/*/eval_score.txt 2>/dev/null

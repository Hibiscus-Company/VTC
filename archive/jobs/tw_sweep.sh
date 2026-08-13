#!/bin/bash
# texture_weight (LEGS-inspired Sobel-weighted L1) eval-split sweep.
# Bracket lambda={1.0, 3.0} on chair (production lpearly recipe) and bonsai (capD/pC recipe),
# on their EXISTING train_sub eval-splits. Score on eval_gt with the exact competition metric.
# Baselines already on record: chair lpearly eval=69.37 (no texture_weight), bonsai capD eval=71.90.
# NO set -u (conda hooks reference unbound vars); pipefail + explicit || checks.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
CHA="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
BON="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"

tw_run() {  # $1 gpu $2 scene $3 tw_lambda  rest=recipe args
  local G=$1 SC=$2 TW=$3; shift 3
  local ES=/mnt/d/avv/evalsplit/$SC M=/mnt/d/avv/tw_test/${SC}_tw${TW}
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --seed 42 --out $M --texture_weight $TW "$@" \
    2>&1 | tail -2 || { echo "TW $SC $TW TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_tw${TW}"
}

( tw_run 0 chair 1.0 $CHA
  tw_run 0 chair 3.0 $CHA ) &
( tw_run 1 bonsai 1.0 $BON
  tw_run 1 bonsai 3.0 $BON ) &
wait
echo "=== TW_SWEEP DONE ==="

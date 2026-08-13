#!/bin/bash
# CONSULT#3 KILL-TESTS (post-gates GPU block, ~5.5h total on 2 GPUs):
#  K1 GPU0: M1 bonsai — DROP UT -> antialiased mode (pinhole k1=0: UT is pure cost).
#           pC recipe minus --ut, train_sub, eval vs 71.36.
#  K2 GPU1: M2 bonsai — init_clip 1.5 (drop mirror-world points) + STANDARD churn.
#           If no collapse (healthy train L1) AND eval >= pC: mechanism confirmed -> scale up.
#  K3 GPU0: M1 chair — lpearly60k minus --ut, eval vs 68.54.
#  K4 GPU1: M6 — seed-spread: bonsai pC with seeds 7,1234 (42 exists) -> winner's-curse noise floor.
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

evalrun() {  # $1 gpu $2 scene $3 name  rest = train args (NOTE: --ut NOT implied here)
  local G=$1 SC=$2 N=$3; shift 3
  local ES=/mnt/d/avv/evalsplit/$SC
  local M=/mnt/d/avv/${SC}_eval/$N
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --seed 42 --out $M "$@" 2>&1 | tail -3
  echo "$*" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py \
    --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_$N"
}

( evalrun 0 bonsai K1_noUT_aa   --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000
  evalrun 0 chair  K3_noUT_aa   --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000 ) &
( evalrun 1 bonsai K2_clip_full --ut --init_clip 1.5 --iters 30000 --cap_max 5000000 --lpips_from 25000
  evalrun 1 bonsai K4_pC_seed7   --ut --seed 7    --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000
  evalrun 1 bonsai K4_pC_seed1k  --ut --seed 1234 --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 ) &
wait
echo "=== KILL TESTS DONE ==="

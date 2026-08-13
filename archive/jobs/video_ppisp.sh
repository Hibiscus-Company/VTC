#!/bin/bash
# LEVER 4 (strategy audit): per-frame exposure/WB (PPISP) on the video scenes, eval-gated.
# Video drift is 25% (vs 2-5% where appearance nulled on towers). Both candidates stack PPISP
# on each scene's eval-winning recipe. Render eval poses WITH --ppisp (controller predicts
# exposure for novel views). exp25b warning respected: activation BEFORE end (default 29/30),
# never render a pre-activation ckpt.
#   GPU0: chair  lpearly60k + ppisp   (vs 68.54 baseline)
#   GPU1: bonsai pC30k     + ppisp   (vs 71.36 baseline)
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

run() {  # $1 gpu  $2 scene  $3 name  rest = train args
  local G=$1 SC=$2 N=$3; shift 3
  local ES=/mnt/d/avv/evalsplit/$SC
  local M=/mnt/d/avv/${SC}_eval/$N
  mkdir -p /mnt/d/avv/${SC}_eval
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --ut --seed 42 --out $M --ppisp "$@" 2>&1 | tail -2
  echo "--ppisp $*" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png \
    --ppisp $M/ppisp.pt --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py \
    --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_$N"
}

( run 0 chair ppisp_lpearly --iters 60000 --cap_max 8000000 \
      --refine_stop 50000 --noise_stop 50000 --lpips_from 30000 ) &
( run 1 bonsai ppisp_pc --iters 30000 --cap_max 5000000 \
      --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 ) &
wait
echo "=== VIDEO PPISP DONE ==="

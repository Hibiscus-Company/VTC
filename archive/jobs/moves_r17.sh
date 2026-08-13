#!/bin/bash
# T2 (round16 members): same proven 1/N mechanism that just delivered +0.232/tower on r15.
#   - 4th UT tower seed (77) on all 5 towers, identical r2r9 recipe -> banked in r2r9/models
#   - 3rd AA video seed (13) for chair + bonsai, identical r14 recipes -> banked in r14/
# GPU0: chair(long) + 2 towers | GPU1: bonsai(short) + 3 towers (~balanced 10-11h each).
# NO set -u (conda hooks reference unbound vars); pipefail + explicit || checks.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
TOW_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"
BON_ARGS="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"
CHA_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"

tower() {  # $1 gpu $2 scene   -- UT seed 77, r2r9 recipe, native render
  local G=$1 s=$2
  local M=/mnt/d/avv/r2r9/models/${s}_ut77
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $PRI/$s/train --images images --ut --seed 77 \
    --out $M $TOW_ARGS 2>&1 | tail -1 || { echo "!!! T2 TOWER $s FAIL"; return 1; }
  echo "$TOW_ARGS" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
    --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
  echo "=== T2 TOWER [$s] DONE ==="
}

video() {  # $1 gpu $2 scene $3 outdir  rest=args   -- NO --ut = antialiased, seed 13
  local G=$1 SC=$2 M=$3; shift 3
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $PRI/$SC/train --images images --seed 13 --out $M "$@" 2>&1 | tail -1 \
    || { echo "!!! T2 VIDEO $SC FAIL"; return 1; }
  echo "noUT-antialiased $*" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $PRI/$SC/test/test_poses.csv \
    --out $M/test_render --png_dir $M/test_png 2>&1 | tail -1
  echo "=== T2 VIDEO [$SC] DONE ==="
}

( video 0 chair /mnt/d/avv/r14/chair_aa13 $CHA_ARGS
  tower 0 HCM0421
  tower 0 HCM0540 ) &
( video 1 bonsai /mnt/d/avv/r14/bonsai_aa13 $BON_ARGS
  tower 1 HCM0539
  tower 1 HCM0644
  tower 1 HCM0674 ) &
wait
echo "=== MOVES_R17 (T2) DONE ==="

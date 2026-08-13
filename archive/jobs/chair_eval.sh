#!/bin/bash
# CHAIR EVAL LOOP, stage 1: baseline = the production recipe (60k/8M, what round11 ships)
# trained on chair train_sub (147), scored on the 58 eval holes. Reference for all chair
# candidates + answers "is chair a hidden laggard" (its field is zeroed; DoF blur).
# Usage: chair_eval.sh <gpu> <name> [extra train args...]
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
G=$1; N=$2; shift 2
ES=/mnt/d/avv/evalsplit/chair
M=/mnt/d/avv/chair_eval/$N
mkdir -p /mnt/d/avv/chair_eval
conda activate gsplat
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
  --source $ES/train_sub --images images --ut --seed 42 --out $M \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 \
  "$@" 2>&1 | tail -2
echo "--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 $*" > $M/train_args.txt
CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
  --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png \
  --ut_render native 2>&1 | tail -1
conda activate fastgs2
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py \
  --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "chair_$N"
echo "=== CHAIR $N DONE ==="

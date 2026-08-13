#!/bin/bash
# Overnight Phase-1 config batch (ZERO code changes, pure flag A/Bs). Waits for tonight's
# queue (absgrad_test.sh, which itself waits for pose_opt v3) to fully clear, then runs.
# All on existing eval-splits with cached baselines: chair 69.37, bonsai 71.90, tower=absgrad0.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

CHA="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
BON="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"
TOW="--ut --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

# video-scene test: $1 gpu $2 scene $3 name  rest=full recipe (overrides)
vid_test() {
  local G=$1 SC=$2 N=$3; shift 3
  local ES=/mnt/d/avv/evalsplit/$SC M=/mnt/d/avv/tw_test/${SC}_$N
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --seed 42 --out $M "$@" 2>&1 | tail -2 \
    || { echo "$SC $N TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_$N"
}
# tower test (UT): $1 gpu $2 name  rest=recipe
tow_test() {
  local G=$1 N=$2; shift 2
  local ES=/mnt/d/avv/evalsplit/HCM0181 M=/mnt/d/avv/tw_test/HCM0181_$N
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --seed 42 --out $M "$@" 2>&1 | tail -2 \
    || { echo "HCM0181 $N TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png \
    --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "HCM0181_$N"
}

# wait for the whole tonight queue to finish
while pgrep -f "poseopt_v3|absgrad_test.sh|HCM0181_absgrad" >/dev/null; do sleep 60; done
echo "=== tonight queue clear, starting overnight Wave-1 config batch $(date) ==="

# GPU0: chair cap_max 8M->4M, then chair noise cooldown 50k->40k
( vid_test 0 chair capmax4M  $CHA --cap_max 4000000
  vid_test 0 chair noise40k  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 40000 --lpips_from 30000
  echo "=== GPU0 overnight batch done ===" ) &
# GPU1: bonsai cap_max 5M->2.5M, then init_clip on tower
( vid_test 1 bonsai capmax2p5M $BON --cap_max 2500000
  tow_test 1 initclip2 $TOW --init_clip 2.0
  echo "=== GPU1 overnight batch done ===" ) &
wait
echo "=== OVERNIGHT_WAVE1 DONE $(date) ==="

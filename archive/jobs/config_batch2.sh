#!/bin/bash
# Phase-1 config A/Bs, RELAUNCH (the overnight version deadlocked: its `pgrep -f poseopt_v3`
# wait-guard matched the MONITOR process's own command line, so it never started). GPUs are
# free now, training is done -> run directly, no wait. Writes a sentinel DONE file at the end
# so the monitor keys off the file, not pgrep (avoids the self-match bug entirely).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
rm -f /mnt/d/avv/config_batch2.DONE

CHA="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
BON="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"
TOW="--ut --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

vid_test() {  # $1 gpu $2 scene $3 name  rest=recipe
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
tow_test() {  # $1 gpu $2 name  rest=recipe
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

echo "=== config_batch2 START $(date) ==="
( vid_test 0 chair capmax4M  $CHA --cap_max 4000000
  vid_test 0 chair noise40k  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 40000 --lpips_from 30000
  echo "=== GPU0 batch done ===" ) &
( vid_test 1 bonsai capmax2p5M $BON --cap_max 2500000
  tow_test 1 initclip2 $TOW --init_clip 2.0
  echo "=== GPU1 batch done ===" ) &
wait
echo "=== CONFIG_BATCH2 DONE $(date) ==="
touch /mnt/d/avv/config_batch2.DONE

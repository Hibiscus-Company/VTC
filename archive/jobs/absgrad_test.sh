#!/bin/bash
# absgrad=True eval-split test on HCM0181 (public set1 tower, cheap proxy -- no private_set2
# tower eval-split exists yet). One-flag experiment, gsplat-native, AbsGS-inspired.
# Waits for whichever GPU frees first from the pose_opt v3 tests, then claims it.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
TOW="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"
ES=/mnt/d/avv/evalsplit/HCM0181

run_ag() {  # $1 gpu $2 use_absgrad(0/1)
  local G=$1 AG=$2
  local M=/mnt/d/avv/tw_test/HCM0181_absgrad${AG}
  local FLAG=""; [[ $AG == 1 ]] && FLAG="--absgrad"
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --ut --seed 42 --out $M $FLAG $TOW \
    2>&1 | tail -2 || { echo "ABSGRAD $AG TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png \
    --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "HCM0181_absgrad${AG}"
}

while pgrep -f "bonsai_poseopt_v3" >/dev/null; do sleep 30; done
echo "=== GPU1 free, starting absgrad=True tower test ==="
run_ag 1 1 &
AGPID=$!

while pgrep -f "chair_poseopt_v3" >/dev/null; do sleep 30; done
echo "=== GPU0 free, starting absgrad=False baseline tower test (same recipe, for a clean A/B) ==="
run_ag 0 0 &
BASEPID=$!

wait $AGPID $BASEPID
echo "=== ABSGRAD_TEST DONE ==="

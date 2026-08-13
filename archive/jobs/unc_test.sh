#!/bin/bash
# uncertainty_weight eval-split test: chair + bonsai, STANDALONE (no pose_opt stacking,
# per Fable's own caution -- keep failure attribution clean). Waits for whichever GPU
# frees first from the pose_opt v2 tests, then claims it.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
CHA="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
BON="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"

unc_run() {  # $1 gpu $2 scene  rest=recipe args
  local G=$1 SC=$2; shift 2
  local ES=/mnt/d/avv/evalsplit/$SC M=/mnt/d/avv/tw_test/${SC}_uncweight
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --seed 42 --out $M \
    --uncertainty_weight --uncertainty_lr 1e-3 --uncertainty_warmup 3000 "$@" \
    2>&1 | tail -2 || { echo "UNC $SC TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_uncweight"
}

# wait for bonsai's GPU (1) to free first (shorter recipe, expected first)
while pgrep -f "bonsai_poseopt_v2" >/dev/null; do sleep 30; done
echo "=== GPU1 free, starting bonsai uncertainty_weight test ==="
unc_run 1 bonsai $BON &
BONPID=$!

while pgrep -f "chair_poseopt_v2" >/dev/null; do sleep 30; done
echo "=== GPU0 free, starting chair uncertainty_weight test ==="
unc_run 0 chair $CHA &
CHAPID=$!

wait $BONPID $CHAPID
echo "=== UNC_TEST DONE ==="

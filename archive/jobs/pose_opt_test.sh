#!/bin/bash
# pose_opt eval-split test: chair + bonsai, production recipes + --pose_opt.
# GPU0 (chair currently free since appaffine test is on GPU1; GPU0 idle).
# Baselines: chair lpearly=69.37, bonsai capD=71.90.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
CHA="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
BON="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"

po_run() {  # $1 gpu $2 scene  rest=recipe args
  local G=$1 SC=$2; shift 2
  local ES=/mnt/d/avv/evalsplit/$SC M=/mnt/d/avv/tw_test/${SC}_poseopt
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --seed 42 --out $M --pose_opt --pose_lr 1e-4 "$@" \
    2>&1 | tail -2 || { echo "POSEOPT $SC TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_poseopt"
}

po_run 0 chair $CHA
echo "=== chair poseopt done, waiting for GPU1 appaffine before bonsai ==="
while pgrep -f "chair_appaffine" >/dev/null; do sleep 30; done
po_run 1 bonsai $BON
echo "=== POSEOPT_TEST DONE ==="

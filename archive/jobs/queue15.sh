#!/bin/bash
# GPU0 pipeline (user: "maxxing use GPU0"):
#   1. Track B smoke: 1k-iter gsplat MCMC mini-train + render 3 test poses
#   2. FastGS ladder e15/e16/e17 (was blocked behind wedged e14 on GPU1)
#   3. private-8 member B (submission critical path)
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
export OUT_ROOT=/mnt/d/avv/output

# --- 1. Track B smoke (gsplat env) ---
conda activate gsplat
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
  --source $S/train --out /mnt/d/avv/output/HCM0181_gsplat_smoke \
  --iters 1000 --cap_max 1000000 2>&1 | grep -E "views|scene_scale|^\[|Saved|Error|Traceback" | tail -15
CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
  --ckpt /mnt/d/avv/output/HCM0181_gsplat_smoke/ckpt.pt \
  --csv $S/test/test_poses.csv --out /mnt/d/avv/output/HCM0181_gsplat_smoke/renders \
  --distort auto --sparse $S/train/sparse/0 2>&1 | tail -2
echo "TRACKB SMOKE DONE"

# --- 2+3. FastGS ladder + member B (fastgs2 env via run_scenes.sh) ---
run1() {
  MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
  EXTRA_ARGS="--lambda_lpips 0.1 $2" \
  bash run_scenes.sh 0 "$1" $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
  echo "=== score $1 ==="
  mkdir -p ~/densq/$1
  ln -sfn $OUT_ROOT/HCM0181_$1/test_poses_renders ~/densq/$1/HCM0181
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub ~/densq/$1 --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
}
run1 e15ceil95  "--metric_gate 1 --opacity_ceiling 0.95"
run1 e16app     "--metric_gate 1 --appearance_affine --lambda_app 0.1"
run1 e17visnorm "--metric_gate 1 --importance_vis_norm --max_gaussians 8000000"
echo "LADDER DONE"

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1" \
bash run_scenes.sh 0 memB \
  $PRI/HCM0249 $PRI/HCM0254 $PRI/HCM0276 $PRI/HCM1439 \
  $PRI/HNI0131 $PRI/HNI0265 $PRI/HNI0366 $PRI/HNI0437 \
  2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
echo "MEMBER B DONE"

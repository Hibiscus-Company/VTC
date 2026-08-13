#!/bin/bash
# SUPERSEDED by queue15.sh (2026-07-13): user opened GPU0 ("maxxing use GPU0"),
# ladder e15-e17 + member B moved there. DO NOT LAUNCH — when queue13/e14 dies,
# GPU1 goes to the Track B full 30k run instead.
exit 1
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2
export OUT_ROOT=/mnt/d/avv/output
rm -rf /mnt/d/avv/output/HCM0181_e14gateoff   # partial dir from killed run

run1() {
  MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
  EXTRA_ARGS="--lambda_lpips 0.1 $2" \
  bash run_scenes.sh 1 "$1" $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
  echo "=== score $1 ==="
  mkdir -p ~/densq/$1
  ln -sfn $OUT_ROOT/HCM0181_$1/test_poses_renders ~/densq/$1/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$1 --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
}
run1 e15ceil95  "--metric_gate 1 --opacity_ceiling 0.95"
run1 e16app     "--metric_gate 1 --appearance_affine --lambda_app 0.1"
run1 e17visnorm "--metric_gate 1 --importance_vis_norm"
echo "LADDER DONE"

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1" \
bash run_scenes.sh 1 memB \
  $PRI/HCM0249 $PRI/HCM0254 $PRI/HCM0276 $PRI/HCM1439 \
  $PRI/HNI0131 $PRI/HNI0265 $PRI/HNI0366 $PRI/HNI0437 \
  2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
echo "MEMBER B DONE"

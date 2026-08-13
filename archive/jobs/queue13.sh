#!/bin/bash
# Post-reboot consolidated queue (GPU1):
#   re-render+score e13gate0 -> train e14gateoff, e15ceil95, e16app, e17visnorm -> member B (private-8)
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2
export OUT_ROOT=/mnt/d/avv/output

# e13: model on C: survived; re-render + score
CUDA_VISIBLE_DEVICES=1 python render_test_poses.py -m output/HCM0181_e13gate0 \
  --csv $S/test/test_poses.csv --out output/HCM0181_e13gate0/test_poses_renders_fixed \
  --mult 0.7 --distort auto --sparse $S/train/sparse/0 \
  --png_dir output/HCM0181_e13gate0/test_poses_renders_png 2>&1 | tail -1
echo "=== score e13gate0 ==="
mkdir -p ~/densq/e13gate0
ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_e13gate0/test_poses_renders_fixed ~/densq/e13gate0/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/e13gate0 --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("

run1() {
  MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
  EXTRA_ARGS="--lambda_lpips 0.1 $2" \
  bash run_scenes.sh 1 "$1" $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
  echo "=== score $1 ==="
  mkdir -p ~/densq/$1
  ln -sfn $OUT_ROOT/HCM0181_$1/test_poses_renders ~/densq/$1/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$1 --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
}
run1 e14gateoff "--metric_gate -1"
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

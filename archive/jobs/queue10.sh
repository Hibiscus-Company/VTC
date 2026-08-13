#!/bin/bash
# Post-submission cheap-lever queue (audit round 2), champion base, HCM0181:
#   exp13: metric_gate 0          (gate curve unsaturated)
#   exp14: metric_gate -1         (gate fully off; only if exp13 model fits VRAM)
#   exp15: opacity_ceiling 0.95   (H3)
#   exp16: appearance_affine      (with recentering)
# Waits for the submission build to finish (marker in queue9.log).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2

until grep -qE "SUBMISSION DONE|FAILED" /home/bkai/.claude/jobs/1c9cf7e9/tmp/queue9.log 2>/dev/null; do sleep 120; done

run() {
  MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
  IMAGES_DIR=images_undist DISTORT=auto \
  EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate 1 $2" \
  bash run_scenes.sh 1 "$1" $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
}
# exp13 overrides the gate (last flag wins is NOT guaranteed -> pass gate explicitly per-exp)
MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate 0" \
bash run_scenes.sh 1 e13gate0 $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"

N13=$(python -c "
from plyfile import PlyData
print(PlyData.read('output/HCM0181_e13gate0/point_cloud/iteration_30000/point_cloud.ply')['vertex'].count)" 2>/dev/null || echo 99999999)
echo "=== e13 gaussians: $N13 ==="
if [ "$N13" -lt 8000000 ]; then
  MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
  EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate -1" \
  bash run_scenes.sh 1 e14gateoff $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
else
  echo "=== SKIP e14: e13 already at $N13 gaussians ==="
fi

run e15ceil95 "--opacity_ceiling 0.95"
run e16app "--appearance_affine --lambda_app 0.1"

for tag in e13gate0 e14gateoff e15ceil95 e16app; do
  [ -d output/HCM0181_$tag/test_poses_renders ] || continue
  echo "=== score $tag ==="
  mkdir -p ~/densq/$tag
  ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_$tag/test_poses_renders ~/densq/$tag/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$tag --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
done
echo "CHEAP LEVERS DONE"

#!/bin/bash
# exp10/exp11: metric_gate dose-response (1 and 3) on g15+lpips base, HCM0181.
# Waits for the lambda sweep to finish by watching queue5.log for its end marker.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2

until grep -q "QUEUE DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/queue5.log 2>/dev/null; do sleep 60; done

run() {
  MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
  IMAGES_DIR=images_undist DISTORT=auto EXTRA_ARGS="--lambda_lpips 0.1 $2" \
  bash run_scenes.sh 1 "$1" $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
}
run e10gate1 "--metric_gate 1"
run e11gate3 "--metric_gate 3"

for tag in e10gate1 e11gate3; do
  echo "=== score $tag ==="
  mkdir -p ~/densq/$tag
  ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_$tag/test_poses_renders ~/densq/$tag/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$tag --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
done
echo "GATE SWEEP DONE"

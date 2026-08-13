#!/bin/bash
# Audit-driven gate experiments on g15+lpips base, HCM0181:
#   exp08: metric_gate 5->2   (H1: densify gate starves thin structures)
#   exp09: final_prune_thresh 0.9->0.98   (H2: final prune deletes hard-region gaussians)
# then exp07 lambda sweep (queue5)
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2

# (wait loop removed — exp06 finished; it also self-deadlocked by matching its
#  own launcher's cmdline, which embedded this script's text via heredoc)

run() {
  MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
  IMAGES_DIR=images_undist DISTORT=auto EXTRA_ARGS="--lambda_lpips 0.1 $2" \
  bash run_scenes.sh 1 "$1" $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error|Gaussian number"
}
run e08gate2 "--metric_gate 2"
run e09prune98 "--final_prune_thresh 0.98"

for tag in e08gate2 e09prune98; do
  echo "=== score $tag ==="
  mkdir -p ~/densq/$tag
  ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_$tag/test_poses_renders ~/densq/$tag/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$tag --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
done
echo "GATES DONE"
bash /home/bkai/.claude/jobs/1c9cf7e9/tmp/queue5.sh

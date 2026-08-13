#!/bin/bash
# exp17: --importance_vis_norm on champion base, HCM0181. Chained after queue10.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2
until grep -q "CHEAP LEVERS DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/queue10.log 2>/dev/null; do sleep 120; done

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate 1 --importance_vis_norm" \
bash run_scenes.sh 1 e17visnorm $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"

echo "=== score e17visnorm ==="
mkdir -p ~/densq/e17visnorm
ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_e17visnorm/test_poses_renders ~/densq/e17visnorm/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/e17visnorm --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
echo "VISNORM DONE"

#!/bin/bash
# exp06: STACK g15 (grad_abs 0.00015) + lpips-ft (lambda 0.1 @25k) on all 5 public scenes
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
P=~/data/phase1/public_set
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1" \
bash run_scenes.sh 1 e06stack $P/HCM0181 $P/HCM0193 $P/HCM0204 $P/hcm0031 $P/hcm0034 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"

echo "=== score e06 FULL SET ==="
mkdir -p ~/densq/e06full
for s in hcm0031 hcm0034 HCM0181 HCM0193 HCM0204; do
  ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/${s}_e06stack/test_poses_renders ~/densq/e06full/$s
done
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/e06full --device cuda:0 --out_json ~/densq/e06full_eval.json 2>&1 | grep -E "n= |Score\(|MEAN"
echo "QUEUE DONE"

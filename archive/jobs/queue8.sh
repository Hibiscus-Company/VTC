#!/bin/bash
# exp12: champion config full-set validation: g15 + lambda_lpips 0.1@25k + metric_gate 1
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
P=~/data/phase1/public_set
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate 1" \
bash run_scenes.sh 1 e12champ $P/HCM0193 $P/HCM0204 $P/hcm0031 $P/hcm0034 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"

echo "=== score e12 FULL SET ==="
mkdir -p ~/densq/e12full
ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_e10gate1/test_poses_renders ~/densq/e12full/HCM0181
for s in HCM0193 HCM0204 hcm0031 hcm0034; do
  ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/${s}_e12champ/test_poses_renders ~/densq/e12full/$s
done
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/e12full --device cuda:0 --out_json ~/densq/e12full_eval.json 2>&1 | grep -E "n= |Score\(|MEAN"
echo "CHAMP DONE"

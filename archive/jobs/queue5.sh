#!/bin/bash
# exp07: lpips-ft sweep around exp04's lambda=0.1@25k, on g15 base, HCM0181
# a: lambda 0.2@25k   b: lambda 0.05@25k   c: lambda 0.1@20k
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2

run() {  # tag extra_args
  MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
  IMAGES_DIR=images_undist DISTORT=auto EXTRA_ARGS="$2" \
  bash run_scenes.sh 1 "$1" $S 2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
}
run e07a_l02 "--lambda_lpips 0.2"
run e07b_l005 "--lambda_lpips 0.05"
run e07c_l01f20 "--lambda_lpips 0.1 --lpips_from_iter 20000"

for tag in e07a_l02 e07b_l005 e07c_l01f20; do
  echo "=== score $tag ==="
  mkdir -p ~/densq/$tag
  ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_$tag/test_poses_renders ~/densq/$tag/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$tag --device cuda:0 2>&1 | grep -E "HCM0181 |Score\(vgg|Score\(alex"
done
echo "QUEUE DONE"

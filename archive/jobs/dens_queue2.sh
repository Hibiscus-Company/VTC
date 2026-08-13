#!/bin/bash
# Follow-up: wait for orphan exp02 training (pid $1), render it, train exp03, score both
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2

while kill -0 "$1" 2>/dev/null; do sleep 15; done
echo "exp02 training (pid $1) finished"
ls output/HCM0181_e02sched40k/point_cloud/ || { echo "FAILED: no exp02 checkpoint"; exit 1; }

CUDA_VISIBLE_DEVICES=1 python render_test_poses.py -m output/HCM0181_e02sched40k \
  --csv $S/test/test_poses.csv --out output/HCM0181_e02sched40k/test_poses_renders \
  --mult 0.7 --distort auto --sparse $S/train/sparse/0 2>&1 | tail -2

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
IMAGES_DIR=images_undist DISTORT=auto \
bash run_scenes.sh 1 e03ga15 $S 2>&1 | grep -vE "Training progress|it/s" | tail -8

for tag in e02sched40k e03ga15; do
  echo "=== score $tag ==="
  mkdir -p ~/densq/$tag && ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_$tag/test_poses_renders ~/densq/$tag/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$tag --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
done
echo "QUEUE DONE"

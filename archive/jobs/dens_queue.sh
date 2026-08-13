#!/bin/bash
# exp02 (40k proportional schedule) + exp03 (grad_abs 0.00015) on HCM0181, GPU1
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.0002 \
IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--iterations 40000 --densify_until_iter 20000 --position_lr_max_steps 40000 --save_iterations 40000 --test_iterations 40000" \
bash run_scenes.sh 1 e02sched40k $S 2>&1 | grep -vE "^Training progress|it/s" | tail -15

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
IMAGES_DIR=images_undist DISTORT=auto \
bash run_scenes.sh 1 e03ga15 $S 2>&1 | grep -vE "^Training progress|it/s" | tail -15

source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2
for tag in e02sched40k e03ga15; do
  echo "=== score $tag ==="
  mkdir -p ~/densq/$tag && rm -f ~/densq/$tag/HCM0181 && ln -sfn /mnt/c/Users/BKAI/an_plaza2/FastGS/output/HCM0181_$tag/test_poses_renders ~/densq/$tag/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$tag --device cuda:0 2>&1 | grep -E "HCM0181|Score\("
  echo "gaussians: $(python -c "
from plyfile import PlyData
import glob
p=sorted(glob.glob('output/HCM0181_$tag/point_cloud/iteration_*/point_cloud.ply'))[-1]
print(p, PlyData.read(p)['vertex'].count)" 2>/dev/null || echo n/a)"
done
echo "QUEUE DONE"

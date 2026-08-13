#!/bin/bash
# Private-8 ensemble member B: g15+lpips, stock gate (e06stack config). Chained after queue11.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2
until grep -qE "VISNORM DONE|FAILED" /home/bkai/.claude/jobs/1c9cf7e9/tmp/queue11.log 2>/dev/null; do sleep 120; done

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1" \
bash run_scenes.sh 1 memB \
  $PRI/HCM0249 $PRI/HCM0254 $PRI/HCM0276 $PRI/HCM1439 \
  $PRI/HNI0131 $PRI/HNI0265 $PRI/HNI0366 $PRI/HNI0437 \
  2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"
echo "MEMBER B DONE"

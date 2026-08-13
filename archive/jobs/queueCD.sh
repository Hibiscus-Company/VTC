#!/bin/bash
# Private members C (gate2) and D (gate0), GPU0, chained after member B.
# Same champion recipe as memB except metric_gate (config-jitter decorrelation
# per audit round 3: A=gate1, B=gate5 stock, C=gate2, D=gate0).
set -o pipefail
until grep -q "MEMBER B DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/queue15.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
export OUT_ROOT=/mnt/d/avv/output

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate 2" \
bash run_scenes.sh 0 memC \
  $PRI/HCM0249 $PRI/HCM0254 $PRI/HCM0276 $PRI/HCM1439 \
  $PRI/HNI0131 $PRI/HNI0265 $PRI/HNI0366 $PRI/HNI0437 \
  2>&1 | grep --line-buffered -vE "Training progress|it/s" | grep --line-buffered -E "===|FAILED|Error"
echo "MEMBER C DONE"

MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate 0" \
bash run_scenes.sh 0 memD \
  $PRI/HCM0249 $PRI/HCM0254 $PRI/HCM0276 $PRI/HCM1439 \
  $PRI/HNI0131 $PRI/HNI0265 $PRI/HNI0366 $PRI/HNI0437 \
  2>&1 | grep --line-buffered -vE "Training progress|it/s" | grep --line-buffered -E "===|FAILED|Error"
echo "MEMBER D DONE"
echo "MEMBERS CD DONE"

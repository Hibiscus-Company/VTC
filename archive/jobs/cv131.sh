#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
PRI=~/data/phase1/private_set1
for s in HNI0131 HNI0265; do
  rd=/mnt/d/avv/output/${s}_gsplatB9ut/train_renders_warp
  [ -d "$rd" ] || { echo "$s: no warp renders yet"; continue; }
  python /home/bkai/.claude/jobs/1c9cf7e9/tmp/d10_privcv.py \
    --render_dir $rd --gt_dir $PRI/$s/train/images --tag $s 2>&1 | tail -1
done
echo "CV131 DONE"

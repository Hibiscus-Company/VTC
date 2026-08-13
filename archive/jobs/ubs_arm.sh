#!/bin/bash
# UBS-6D on bonsai eval split. Waits for GPU1 to free (production seed 777 finishes ~00:55).
# Recipe matched to ours where it matters: VGG-LPIPS lambda=0.1 from 12k (patched into their
# train.py), cap_max 5M, 30k iters. Their defaults keep input_dim=6 (spatial+angular Beta).
# THRESHOLD: >= +1.0 over 71.799 to count. The eval split has sign-inverted TWICE when comparing
# ACROSS model classes, and UBS is a different model class.
source ~/miniconda3/etc/profile.d/conda.sh
T=/home/bkai/.claude/jobs/1c9cf7e9/tmp; ES=/mnt/d/avv/evalsplit/bonsai
OUT=/mnt/d/avv/r39_ubs; mkdir -p $OUT
while [ "$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]; do sleep 60; done
conda activate ubs
cd $T/ubs_repo
echo ">>> UBS-6D cap5M (gpu1) START $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=1 python train.py -s /mnt/d/avv/ubs_data -m $OUT/cap5M \
  --iterations 30000 --cap_max 5000000 --eval 2>&1 | tail -12
PLY=$OUT/cap5M/point_cloud/iteration_30000/point_cloud.ply
[ -f "$PLY" ] || PLY=$(ls -t $OUT/cap5M/point_cloud/*/point_cloud.ply 2>/dev/null|head -1)
echo "ply: $PLY"
[ -f "$PLY" ] || { echo "!!! UBS NO PLY"; touch $OUT/FAILED; exit 1; }
CUDA_VISIBLE_DEVICES=1 python $T/ubs_render.py "$PLY" $ES/eval_poses.csv $OUT/cap5M/eval_png 2>&1|tail -3
conda activate fastgs2
python $T/bar.py $OUT/cap5M/eval_png
echo "<<< UBS DONE $(date +%H:%M)  [threshold: >= 72.80 to count]"
touch $OUT/cap5M.DONE

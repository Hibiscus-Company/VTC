#!/bin/bash
# Track B smoke rerun (dtype fix). Runs alongside early-phase e15 on GPU0.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
  --source $S/train --out /mnt/d/avv/output/HCM0181_gsplat_smoke \
  --iters 1000 --cap_max 1000000 2>&1 | grep -E "views|scene_scale|^\[|Saved|Error|Traceback|RuntimeError" | tail -12
CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
  --ckpt /mnt/d/avv/output/HCM0181_gsplat_smoke/ckpt.pt \
  --csv $S/test/test_poses.csv --out /mnt/d/avv/output/HCM0181_gsplat_smoke/renders \
  --distort auto --sparse $S/train/sparse/0 2>&1 | tail -3
echo "TRACKB SMOKE2 DONE"

#!/bin/bash
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONUNBUFFERED=1
PRI=/mnt/d/avv/data/phase1/private_set2; M=/mnt/d/avv/r17/chair_depth_seed42
conda activate gsplat
# precompute depth for FULL chair train (not eval-split subset)
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_depth.py --images $PRI/chair/train/images \
  --out /mnt/d/avv/depth_prior/chair_full --device cuda:0 2>&1 | tail -1
echo "=== chair depth-prior PRODUCTION $(date) ==="
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py --source $PRI/chair/train --images images \
  --seed 42 --out $M --depth_prior 0.05 --depth_dir /mnt/d/avv/depth_prior/chair_full \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000 \
  2>&1 | tail -2 || { echo "!!! CHAIR DEPTH PROD FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $PRI/chair/test/test_poses.csv --out $M/test_render --png_dir $M/test_png 2>&1 | tail -1
echo "=== CHAIR DEPTH PROD DONE $(date), $(ls $M/test_png|wc -l) renders ==="
touch /mnt/d/avv/chair_depth_prod.DONE

#!/bin/bash
# Follow-up to r17's flat LB result: r17 diluted a real eval-split win (+0.435) through 1-of-3
# chair members AND 1-of-7 scenes -> net +0.0024, inside noise. EMA-ify the OTHER 2 chair seeds
# too so the full 3-seed ensemble carries the improvement, not just 1 slot. This one takes
# whichever GPU frees first from the HCM0421 tower-decay test -- GPU1 (decay0.999 arm).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
M=/mnt/d/avv/r17/chair_ema099_seed7
CHA_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"

while [ ! -f /mnt/d/avv/tower_ema_production.DONE ]; do sleep 30; done
echo "=== GPU1 free (tower decay0.999 done), starting chair_aa7 EMA $(date) ==="
conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $PRI/chair/train --images images --seed 7 --ema_decay 0.99 \
  --out $M $CHA_ARGS 2>&1 | tail -3 || { echo "!!! chair_aa7 EMA TRAIN FAIL"; exit 1; }
echo "--seed 7 --ema_decay 0.99 $CHA_ARGS" > $M/train_args.txt
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $M/ckpt.pt --csv $PRI/chair/test/test_poses.csv \
  --out $M/test_render --png_dir $M/test_png 2>&1 | tail -1
echo "=== chair_aa7 EMA DONE $(date), $(ls $M/test_png | wc -l) test renders ready ==="
touch /mnt/d/avv/chair_aa7_ema_production.DONE

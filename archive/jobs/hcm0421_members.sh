#!/bin/bash
# Build an ENSEMBLE-MATCHED eval set for a real set2 tower (HCM0421): 2 more decorrelated seeds to
# join the seed-42 member the pool is already making, giving a 3-member eval ensemble that mirrors
# the HCM0181 setup where restoration measured +0.315.
#
# WHY this instead of 1 member x 5 towers: the restorer will be applied to 6-7 member PRODUCTION
# ensembles. A restorer trained on single-member renders learns to undo much LARGER noise than an
# ensemble has, and would over-correct. Matching the INPUT DISTRIBUTION beats covering more scenes,
# and cross-scene transfer (HCM0181-trained -> HCM0421) is exactly what this lets us test.
# $1 = seed, $2 = gpu
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
S=$1 G=$2
ES=/mnt/d/avv/evalsplit/HCM0421
M=/mnt/d/avv/evalgen/HCM0421_s${S}
conda activate gsplat
# never stack: wait for this gpu to actually be free
while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 1500 ]; do sleep 60; done
echo "=== HCM0421 seed $S on gpu$G $(date) ==="
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed $S --ut --out $M --ema_decay 0.999 \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 \
  | tail -2 || { echo "!!! s$S TRAIN FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1 \
  || { echo "!!! s$S RENDER FAIL"; exit 1; }
conda activate fastgs2
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt \
  --tag HCM0421_s${S}
rm -f $M/ckpt.pt
echo "=== HCM0421 s$S DONE $(date) ==="
touch /mnt/d/avv/evalgen/HCM0421_s${S}.DONE

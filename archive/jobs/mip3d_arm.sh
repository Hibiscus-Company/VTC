#!/bin/bash
# Mip-Splatting 3D filter DURING training -- one arm. $1=factor $2=gpu
# PAIRED A/B: same seed 101, same production recipe as the seed-101 bank run, whose solo
# score on this exact proxy was 74.9974 (and seed-202 74.9301, so seed noise ~0.03).
# Any move beyond ~+/-0.1 is real signal, not seed jitter.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
MF=$1 G=$2
ES=/mnt/d/avv/evalsplit/HCM0181
M=/mnt/d/avv/mip3d/HCM0181_mf${MF}
mkdir -p /mnt/d/avv/mip3d
conda activate gsplat
echo "=== mip3d factor $MF on gpu$G $(date) (paired baseline seed101 = 74.9974) ==="
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed 101 --ut --out $M --mip3d $MF --mip3d_every 100 \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 \
  | grep -aE "^\[|Baked|Saved|Error|Traceback" | tail -6 || { echo "!!! MF$MF TRAIN FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
  --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1 \
  || { echo "!!! MF$MF RENDER FAIL"; exit 1; }
conda activate fastgs2
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt \
  --tag HCM0181_mip${MF}
rm -f $M/ckpt.pt
echo "=== MF$MF DONE $(date) (baseline seed101 74.9974) ==="
touch /mnt/d/avv/mip3d_mf${MF}.DONE

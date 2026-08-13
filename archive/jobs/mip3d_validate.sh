#!/bin/bash
# CROSS-SCENE VALIDATION of the mip3d filter before committing 10 GPU-h to productionising it.
# HCM0181 gave +0.6596 paired (all three metrics up). HCM0421 is a REAL set2 tower and already has
# a paired baseline from the evalgen pool: seed42 + ema999, 60k/8M, SCORE 75.7616.
# Same seed, same ema, same recipe, only --mip3d added => clean paired A/B on a second scene.
# $1 = gpu
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
G=${1:-0}
ES=/mnt/d/avv/evalsplit/HCM0421
M=/mnt/d/avv/mip3d/HCM0421_mip0.2
while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 2000 ]; do sleep 60; done
conda activate gsplat
echo "=== HCM0421 + mip3d 0.2 on gpu$G $(date) (paired baseline seed42/ema999 = 75.7616) ==="
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed 42 --ut --out $M --ema_decay 0.999 --mip3d 0.2 --mip3d_every 100 \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 \
  | grep -aE "^\[59|Baked|Saved" | tail -3 || { echo "!!! FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
conda activate fastgs2
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt \
  --tag HCM0421_mip0.2
echo "=== baseline HCM0421_evalgen was: PSNR 25.1137 SSIM 0.8403 LPIPS 0.1129 SCORE 75.7616 ==="
rm -f $M/ckpt.pt
touch /mnt/d/avv/mip3d_validate2.DONE
echo "=== MIP3D HCM0421 VALIDATION DONE $(date) ==="

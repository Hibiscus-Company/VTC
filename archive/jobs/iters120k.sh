#!/bin/bash
# THE FIDELITY AXIS: 60k -> 120k iterations. NEVER TESTED.
#
# WHY THIS IS THE RIGHT TARGET NOW:
#  - Today's top-1 decomposition says the 4.9-point gap is FIDELITY-shaped: at PSNR 31-32 / SSIM .92
#    top-1's LPIPS is about equal to ours. They out-reconstruct us, they don't out-perceive us.
#  - Our own log: TRAIN PSNR is only ~27dB => we are UNDERFIT, not overfit. Even perfect
#    generalisation caps us at 27.
#  - And train fit TRANSFERS: d(test)/d(train) = +0.80 dB/dB, r=+0.928, gap constant at 2.03dB.
#  - The schedule knob was STILL DELIVERING when we stopped: 30k->60k = +0.32 (exp29). 60k became
#    "the production recipe" and nobody ever tried further. Loss, regularisation and capacity were
#    each eliminated as the cause of the 27dB wall -- schedule was never eliminated, just abandoned.
#
# Paired against the 60k+mip3d run on the SAME seed/scene (75.6570), and it tests the exact recipe
# we would ship, so a positive result is directly productionisable.
# Schedule params scale with the run: 50k/50k/50k at 60k -> 100k/100k/100k at 120k.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
G=${1:-1}
ES=/mnt/d/avv/evalsplit/HCM0181
M=/mnt/d/avv/iters/HCM0181_mip120k
mkdir -p /mnt/d/avv/iters
while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 2000 ]; do sleep 60; done
conda activate gsplat
echo "=== HCM0181 mip3d + 120k iters on gpu$G $(date) ==="
echo "=== paired baselines: plain60k 74.9974 | mip3d60k 75.6570 (same seed 101) ==="
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed 101 --ut --out $M --mip3d 0.2 --mip3d_every 100 \
  --iters 120000 --cap_max 8000000 --refine_stop 100000 --noise_stop 100000 --lpips_from 100000 2>&1 \
  | grep -aE "^\[1[01][0-9]000|^\[119|Baked|Saved" | tail -5 || { echo "!!! 120k FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
conda activate fastgs2
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt \
  --tag HCM0181_mip120k
echo "=== compare: plain60k 74.9974 | mip3d60k 75.6570 | mip3d120k above ==="
rm -f $M/ckpt.pt
touch /mnt/d/avv/iters120k.DONE

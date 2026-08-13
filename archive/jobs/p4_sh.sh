#!/bin/bash
# P4: SH-degree clamp at render. At 11.8 deg extrapolation from the nearest train view,
# high-order view-dependent colour can hallucinate shading the test pose never justified
# -- a classic silent PSNR tax. Never measured. Costs minutes.
# Baseline = exp29 (HCM0181 60k/8M UT, the best single model).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
CK=/mnt/d/avv/output/HCM0181_gsplatB11ut60k/ckpt.pt
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1

for d in 0 1 2 3; do
  OUT=/mnt/d/avv/output/HCM0181_sh$d
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
    --ckpt $CK --csv $S/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    --ut_render native --sh_degree $d 2>&1 | tail -1 \
    || { echo "=== SH$d RENDER FAILED ==="; continue; }
  conda activate fastgs2
  mkdir -p ~/densq/sh$d && ln -sfn $OUT/test_poses_renders ~/densq/sh$d/HCM0181
  CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub ~/densq/sh$d --device cuda:0 2>&1 \
    | grep -E "Score\(vgg\)" | sed "s/^/P4 sh=$d /"
done
echo "P4 DONE"

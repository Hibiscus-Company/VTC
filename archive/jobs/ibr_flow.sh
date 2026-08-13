#!/bin/bash
# IBR arm 4: flow-corrected (last shot before closing task #12).
# Diagnostic showed stochastic misregistration (no systematic offset);
# DIS flow snaps warped photos onto the splat render before gating.
# Chains after ppft smoke on GPU0. Tight-ish gates + flow_max 3px.
set -o pipefail
until grep -q "PPFT AB DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/ppft_ab.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
CKPT=/mnt/d/avv/output/HCM0181_gsplatB10ut8M/ckpt.pt
OUT=/mnt/d/avv/ibr/HCM0181_flow
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
CUDA_VISIBLE_DEVICES=0 python gsplat_track/ibr_render.py \
  --ckpt $CKPT --source $S/train --csv $S/test/test_poses.csv \
  --out $OUT/renders --png_dir $OUT/renders_png \
  --flow_correct --tau_geo 0.02 --tau_pho 0.05 \
  || { echo "=== IBR FLOW FAILED ==="; exit 1; }
conda activate fastgs2
mkdir -p ~/densq/ibr_flow
ln -sfn $OUT/renders ~/densq/ibr_flow/HCM0181
CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub ~/densq/ibr_flow --device cuda:0 2>&1 \
  | grep -E "HCM0181 |Score\(vgg\)" | sed 's/^/IBR flow /'
echo "IBR FLOW DONE"

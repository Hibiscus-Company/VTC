#!/bin/bash
# exp31 = IDEA 3, METRIC-EXACT FINETUNE. The diagnostics (D1/D2/D3) proved the
# gap is neither appearance (+0.09dB oracle) nor photo-reuse (9.4dB paste) but
# RECONSTRUCTION — and our loss never optimized PSNR at all (L1 = median).
# Exact score-matched loss: 0.4*LPIPS + 0.3*(1-SSIM) + 0.02606*ln(MSE).
# Warm-start from exp29 (best single, 60k/8M UT, 75.8989 / PSNR 24.4646),
# finetune 8k steps, NO densification/noise (pure refit), means-lr x0.3.
# Chains after privUTs7 on GPU1 (before the UT60k half — this outranks it now).
set -o pipefail
until grep -q "PRIVUTS7 DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/privUTs7.log 2>/dev/null; do sleep 60; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
BASE=/mnt/d/avv/output/HCM0181_gsplatB11ut60k/ckpt.pt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat

run_m() {  # name, extra args
  local NAME=$1; shift
  local OUT=/mnt/d/avv/output/HCM0181_$NAME
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
    --source $S/train --images images --ut --init_ckpt $BASE \
    --out $OUT --cap_max 8000000 --refine_stop 0 --noise_stop 0 \
    --opacity_reg 0 --scale_reg 0 --means_lr 4.8e-5 "$@" \
    || { echo "=== $NAME TRAIN FAILED ==="; return 1; }
  echo "=== $NAME TRAIN DONE ==="
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  mkdir -p ~/densq/$NAME
  ln -sfn $OUT/test_poses_renders ~/densq/$NAME/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$NAME --device cuda:0 2>&1 \
    | grep -E "HCM0181 |Score\(vgg\)" | sed "s/^/METRIC $NAME /"
  conda activate gsplat
  echo "=== $NAME SCORED ==="
}

# arm A: exact metric loss (lpips weight 0.4, active from step 0)
run_m ut60kMetric --iters 8000 --metric_loss --lambda_lpips 0.4 --lpips_from -1

# arm B: metric loss WITHOUT lpips (pure PSNR+SSIM push) — isolates how much
# PSNR the log-MSE term can buy, and what LPIPS costs in PSNR
run_m ut60kMetricNoLp --iters 8000 --metric_loss --lambda_lpips 0

echo "METRIC FT DONE"

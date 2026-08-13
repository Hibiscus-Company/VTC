#!/bin/bash
# exp27 = idea 2 (3DGUT): train in DISTORTED space on original train/images
# (VIRGIN pixels — removes the undistort INTER_CUBIC resample generation),
# classic + with_ut + with_eval3d + radial k1; render natively, no warp.
# Audit round-7 pre-flight: MCMC ops confirmed pure-3D (UT-compatible);
# EV caution: classic mode surrenders antialiased opacity compensation.
# Smokes gate the full run:
#   S1: 3k-iter UT mini-train learns (loss trajectory printed)
#   S2: native-vs-warp render agreement (corners diverge if radial_coeffs
#       were silently ignored) + PSNR vs real test GT (convention gate)
# Chains after trackb6 (exp26) on GPU1.
set -o pipefail
until grep -q "TRACKB6 QUEUE DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/trackb6.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
SM=/mnt/d/avv/output/HCM0181_ut_smoke
CSV2=/home/bkai/.claude/jobs/1c9cf7e9/tmp/smoke_poses.csv
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat

head -3 $S/test/test_poses.csv > $CSV2
echo "=== UT SMOKE: 3k mini-train (timing) ==="
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "UT_SMOKE_TRAIN_SECONDS %e" \
python gsplat_track/train_gsplat.py \
  --source $S/train --images images --ut \
  --out $SM --iters 3000 --cap_max 1000000 \
  || { echo "=== UT SMOKE TRAIN FAILED ==="; exit 1; }
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $SM/ckpt.pt --csv $CSV2 --out $SM/native --png_dir $SM/native_png \
  --ut_render native || { echo "=== UT SMOKE NATIVE RENDER FAILED ==="; exit 1; }
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $SM/ckpt.pt --csv $CSV2 --out $SM/warp --png_dir $SM/warp_png \
  --ut_render warp --distort auto --sparse $S/train/sparse/0 \
  || { echo "=== UT SMOKE WARP RENDER FAILED ==="; exit 1; }
python - "$SM" "$S/test/images" <<'EOF' || { echo "=== UT SMOKE CHECK FAILED ==="; exit 1; }
import sys, os, numpy as np
from PIL import Image
sm, gtdir = sys.argv[1], sys.argv[2]
names = sorted(os.listdir(os.path.join(sm, "native_png")))
ok = True
for n in names:
    a = np.asarray(Image.open(os.path.join(sm, "native_png", n)), dtype=np.float32)
    b = np.asarray(Image.open(os.path.join(sm, "warp_png", n)), dtype=np.float32)
    H, W = a.shape[:2]
    cy, cx = H // 4, W // 4
    center = np.abs(a[cy:-cy, cx:-cx] - b[cy:-cy, cx:-cx]).mean()
    ch, cw = H // 10, W // 10
    corner = np.mean([np.abs(a[:ch,:cw]-b[:ch,:cw]).mean(), np.abs(a[:ch,-cw:]-b[:ch,-cw:]).mean(),
                      np.abs(a[-ch:,:cw]-b[-ch:,:cw]).mean(), np.abs(a[-ch:,-cw:]-b[-ch:,-cw:]).mean()])
    gt_name = os.path.splitext(n)[0] + ".JPG"
    gp = os.path.join(gtdir, gt_name)
    psnr = float("nan")
    if os.path.exists(gp):
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32)
        mse = ((a - g) ** 2).mean()
        psnr = 10 * np.log10(255.0 ** 2 / mse)
    print(f"UT_SMOKE {n}: center|nat-warp| {center:.2f}/255  corner {corner:.2f}/255  PSNRvsGT {psnr:.2f}dB")
    if corner > 12.0:
        print(f"UT_SMOKE FAIL: corner divergence {corner:.2f} > 12 — radial_coeffs likely ignored in one path")
        ok = False
    if psnr == psnr and psnr < 15.0:
        print(f"UT_SMOKE FAIL: PSNR {psnr:.2f} < 15dB — convention bug in distorted-space path")
        ok = False
print("UT_SMOKE " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
EOF
echo "=== UT SMOKE PASSED ==="

NAME=gsplatB9ut
OUT=/mnt/d/avv/output/HCM0181_$NAME
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $S/train --images images --ut \
  --out $OUT --iters 30000 --cap_max 5000000 --noise_stop 25000 \
  || { echo "=== $NAME TRAIN FAILED ==="; exit 1; }
echo "=== $NAME TRAIN DONE ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
  --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
  --ut_render native 2>&1 | tail -2
conda activate fastgs2
mkdir -p ~/densq/$NAME
ln -sfn $OUT/test_poses_renders ~/densq/$NAME/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$NAME --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
echo "=== $NAME SCORED ==="
echo "TRACKB7 QUEUE DONE"

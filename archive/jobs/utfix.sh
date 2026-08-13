#!/bin/bash
# Audit round-8 A-1/A-2 remediation: HNI0131 + HNI0265 have k1 ~ -0.115
# (negative -> forward distortion FOLDS at r_u=1.704; 13k-17k op>0.5 gaussians
# project into frame via the folded branch in the UT-native path). Re-render
# both scenes through the (fixed) warp fallback -- pinhole UT render (with_ut+
# with_eval3d kept, radial_coeffs None -> no fold) + analytic DistortionWarp --
# and swap them into the canonical dirs ens5 consumes. Native renders kept as
# *_native for comparison. Runs on GPU1 right after privUT_gpu1 finishes
# (~02:10), hours before ens5 fires.
set -o pipefail
until grep -q "PRIVUT GPU1 DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/privUT_gpu1.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
for s in HNI0131 HNI0265; do
  OUT=/mnt/d/avv/output/${s}_gsplatB9ut
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
    --out $OUT/renders_warp --png_dir $OUT/renders_warp_png \
    --ut_render warp --distort auto --sparse $PRI/$s/train/sparse/0 \
    || { echo "=== utfix $s WARP RENDER FAILED (native renders left in place) ==="; continue; }
  python - "$OUT" <<'EOF'
import sys, os, numpy as np
from PIL import Image
out = sys.argv[1]
names = sorted(os.listdir(os.path.join(out, "test_poses_renders_png")))[:5]
cs, ks = [], []
for n in names:
    a = np.asarray(Image.open(os.path.join(out, "test_poses_renders_png", n)), dtype=np.float32)
    b = np.asarray(Image.open(os.path.join(out, "renders_warp_png", n)), dtype=np.float32)
    H, W = a.shape[:2]; ch, cw = H // 8, W // 8
    ks.append(np.mean([np.abs(a[:ch,:cw]-b[:ch,:cw]).mean(), np.abs(a[:ch,-cw:]-b[:ch,-cw:]).mean(),
                       np.abs(a[-ch:,:cw]-b[-ch:,:cw]).mean(), np.abs(a[-ch:,-cw:]-b[-ch:,-cw:]).mean()]))
    cs.append(np.abs(a[H//4:-H//4, W//4:-W//4] - b[H//4:-H//4, W//4:-W//4]).mean())
print(f"UTFIX {os.path.basename(out)}: native-vs-warp center {np.mean(cs):.2f}/255 corner {np.mean(ks):.2f}/255 (n=5)")
EOF
  mv $OUT/test_poses_renders $OUT/test_poses_renders_native
  mv $OUT/test_poses_renders_png $OUT/test_poses_renders_native_png
  mv $OUT/renders_warp $OUT/test_poses_renders
  mv $OUT/renders_warp_png $OUT/test_poses_renders_png
  echo "=== utfix $s SWAPPED to warp renders ==="
done
echo "UTFIX DONE"

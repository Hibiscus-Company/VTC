#!/bin/bash
# exp31b = THE LAST REAL LEVER, run cheaply this time.
#
# Our loss has NEVER optimized PSNR. Standard 3DGS trains 0.8*L1 + 0.2*(1-SSIM), and
# L1 optimizes the MEDIAN, not the mean. The metric-exact loss is, exactly,
#   0.4*LPIPS + 0.3*(1-SSIM) + 0.02606*ln(MSE)      [0.02606 = 0.06/ln10]
# minimizing which MAXIMIZES the competition score.
#
# exp31 (the first attempt) was killed after 8h33m for 8k steps -- MY design error:
# --lpips_from -1 ran VGG-LPIPS on EVERY step at 8M gaussians (~3.9 s/step). LPIPS belongs
# in the TAIL, as every other recipe we have does it.
#
# Arm B first (NO LPIPS at all): 5-10x faster, and it ISOLATES exactly what the log-MSE term
# buys in PSNR without LPIPS fighting it for pixel alignment. That is the scientific question.
# Arm A second (LPIPS in the tail only) is the shippable variant.
#
# Warm-start exp29 (60k/8M UT, best single = 75.8989 / PSNR 24.4646), 8k-step pure refit,
# no densification, no noise, no reg -- so the ONLY variable is the loss.
#
# Chains behind R9 so it never competes with the members for GPU.
set -o pipefail
until grep -q "R9 ZIP DONE\|R9 ZIP BUILD FAILED\|R9 TIMED OUT" \
  /home/bkai/.claude/jobs/1c9cf7e9/tmp/r9.log 2>/dev/null; do sleep 300; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
BASE=/mnt/d/avv/output/HCM0181_gsplatB11ut60k/ckpt.pt
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1     # exp31's progress was invisible for 8h without this

run() {   # $1 name, rest = extra args
  local NAME=$1; shift
  local OUT=/mnt/d/avv/output/HCM0181_$NAME
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
    --source $S/train --images images --ut --init_ckpt $BASE \
    --out $OUT --iters 8000 --cap_max 8000000 --refine_stop 0 --noise_stop 0 \
    --opacity_reg 0 --scale_reg 0 --means_lr 4.8e-5 --metric_loss "$@" \
    || { echo "=== $NAME TRAIN FAILED ==="; return 1; }
  echo "=== $NAME TRAINED ==="
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    --ut_render native 2>&1 | tail -1
  # TRAIN-view PSNR too: the audit + the consult both called this the single most
  # informative free number. If arm B lifts TRAIN psnr a lot, the loss shape was a real
  # binding constraint; if train barely moves, the cap is elsewhere.
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_train.py \
    --ckpt $OUT/ckpt.pt --source $S/train --images images \
    --out $OUT/train_png --stride 8 --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  python - <<PY
import numpy as np, os
from PIL import Image
def psnr(d,g):
    st={os.path.splitext(f)[0]:f for f in os.listdir(g)}
    v=[]
    for f in sorted(os.listdir(d)):
        s=os.path.splitext(f)[0]
        if s not in st: continue
        a=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"),dtype=np.float32)/255
        b=np.asarray(Image.open(os.path.join(g,st[s])).convert("RGB"),dtype=np.float32)/255
        if a.shape!=b.shape: continue
        v.append(10*np.log10(1/max(float(((a-b)**2).mean()),1e-12)))
    return float(np.mean(v)) if v else float("nan")
print(f"METRIC $NAME TRAIN-PSNR {psnr('$OUT/train_png', os.path.expanduser('~/data/phase1/public_set/HCM0181/train/images')):.4f}")
PY
  mkdir -p ~/densq/$NAME && ln -sfn $OUT/test_poses_renders ~/densq/$NAME/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$NAME --device cuda:0 2>&1 \
    | grep -E "HCM0181 |Score\(vgg\)" | sed "s/^/METRIC $NAME /"
  echo "=== $NAME SCORED ==="
}

# baseline for reference: exp29 = 75.8989, test PSNR 24.4646, train PSNR ~27 (D2 era)
run m31b_nolpips  --lambda_lpips 0                      # ISOLATES the log-MSE term
run m31b_taillpips --lambda_lpips 0.4 --lpips_from 6000 # shippable variant
echo "EXP31B DONE"

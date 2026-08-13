#!/bin/bash
# THE DECISIVE EXPERIMENT (audit r12): what caps our 3DGS TRAIN fit at ~27 dB?
#
# The whole "86 is sealed" argument rests on D5: perfect registration -> only 28.6 dB.
# But D5's dense-flow oracle only REPROJECTS the existing render's pixels; it cannot
# sharpen soft content or fix a wrong surface. So it bounds REGISTRATION, not model
# quality. 27 dB train with 8M gaussians is LOW (vanilla 3DGS reaches 32-35). If the
# model CAN fit train to 32+, registration was never the wall and there's a multi-dB path.
#
# exp31b (warm-start refit, densification OFF) got train only 27.09 -> 27.61 with pure
# metric loss -- but a refit CANNOT add/replace gaussians, so it can't test capacity.
# This is FROM SCRATCH with densification ON, three arms isolating the cause:
#
#   A. standard recipe (control)                 -> reproduces the ~27 dB
#   B. pure-L2 loss, reg ON                       -> does the LOSS cap train fit?
#   C. pure-L2 loss, reg OFF (opacity/scale 0,     -> does REGULARIZATION cap it?
#      noise_stop early)                             (round-6: opacity_reg drains thin structure)
#
# All on HCM0181, 60k/8M UT, TRAIN PSNR is the readout. Runs both GPUs.
# VERDICT: any arm reaching train >=32 => model was under-fit, real path exists.
#          all arms cap <=29 => content is the wall, 86 is sealed, switch to rank.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1

train_eval() {  # $1 gpu, $2 name, rest = train args
  local G=$1 NAME=$2; shift 2
  local OUT=/mnt/d/avv/output/HCM0181_$NAME
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $S/train --images images --ut \
    --out $OUT --iters 60000 --cap_max 8000000 "$@" \
    || { echo "=== $NAME TRAIN FAILED ==="; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_train.py \
    --ckpt $OUT/ckpt.pt --source $S/train --images images \
    --out $OUT/train_png --stride 8 --ut_render native 2>&1 | tail -1
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
    --out $OUT/tp --png_dir $OUT/tp_png --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  python - <<PY
import numpy as np, os
from PIL import Image
def psnr(d,g):
    st={os.path.splitext(f)[0]:f for f in os.listdir(g)}; v=[]
    for f in sorted(os.listdir(d)):
        s=os.path.splitext(f)[0]
        if s not in st: continue
        a=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"),dtype=np.float32)/255
        b=np.asarray(Image.open(os.path.join(g,st[s])).convert("RGB"),dtype=np.float32)/255
        if a.shape!=b.shape: continue
        v.append(10*np.log10(1/max(float(((a-b)**2).mean()),1e-12)))
    return float(np.mean(v)) if v else float("nan")
tr=psnr('$OUT/train_png', os.path.expanduser('~/data/phase1/public_set/HCM0181/train/images'))
te=psnr('$OUT/tp', os.path.expanduser('~/data/phase1/public_set/HCM0181/test/images'))
print(f"FIDFIT $NAME  TRAIN {tr:.4f}  TEST {te:.4f}  gap {tr-te:.4f}")
PY
  echo "=== $NAME DONE ==="
}

# GPU0: control + pure-L2 reg-off (the two most informative)
( train_eval 0 fid_ctrl  --refine_stop 50000 --noise_stop 50000 --lpips_from 50000
  train_eval 0 fid_l2off --pure_l2 --refine_stop 50000 --noise_stop 25000 \
             --opacity_reg 0 --scale_reg 0 ) &
# GPU1: pure-L2 reg-on
( train_eval 1 fid_l2reg --pure_l2 --refine_stop 50000 --noise_stop 50000 ) &
wait
echo "FIDFIT DONE"

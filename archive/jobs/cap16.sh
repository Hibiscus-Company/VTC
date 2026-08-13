#!/bin/bash
# CAPACITY TEST: does more gaussians raise train fit on a DENSE scene, and does it TRANSFER?
#
# Per-scene train PSNR revealed the method is NOT capped at 27: HCM1439 (103 imgs) fits train
# to 32.4 dB, while the dense 240-image scenes fit to 26-27. That is a CAPACITY signal --
# 8M gaussians spread thinner across more views. HCM0181 is dense (240 imgs, train 26.99 @ 8M).
#
# Test: cap_max 16M on HCM0181, standard recipe. Readout = TRAIN and TEST PSNR.
#   train rises AND test follows -> capacity is a real transferable lever, push it on dense scenes
#   train rises, test flat        -> generalization wall, capacity only overfits
#   train flat                    -> not capacity; the 27 cap is something else
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
G=${1:-1}

for CAP in 16000000; do
  NAME=cap$((CAP/1000000))M
  OUT=/mnt/d/avv/output/HCM0181_$NAME
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $S/train --images images --ut \
    --out $OUT --iters 60000 --cap_max $CAP \
    --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 \
    || { echo "=== $NAME TRAIN FAILED (OOM? retry 12M) ==="; exit 1; }
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
print(f"CAP16 $NAME  N={len(os.listdir('$OUT/tp_png'))} TRAIN {tr:.4f}  TEST {te:.4f}  gap {tr-te:.4f}")
PY
  mkdir -p ~/densq/$NAME && ln -sfn $OUT/tp ~/densq/$NAME/HCM0181
  CUDA_VISIBLE_DEVICES=$G python score_submission.py --sub ~/densq/$NAME \
    --gt_root ~/data/phase1/public_set --device cuda:0 2>&1 | grep -E "Score\(vgg\)" | sed "s/^/CAP16 $NAME /"
done
echo "CAP16 DONE"

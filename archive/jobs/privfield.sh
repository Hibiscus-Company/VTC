#!/bin/bash
# R7 = R6 ensemble renders + per-scene TRAIN-fitted lens field. NO retraining.
#
# The field is fit from the gsplatB9ut model's TRAIN renders vs that scene's TRAIN
# photos -- the SAME model fieldval validated on public, so what ships is what was
# validated. Never touches test GT.
#
# Per-scene, NOT pooled: residual_s = (true lens) - (that scene's COLMAP k1), and
# k1 varies per scene (+0.008..+0.014, and -0.115 for HNI0131/HNI0265).
#
# LAUNCH ONLY AFTER fieldval shows 5/5 public positive.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
FLD=/mnt/d/avv/fields
ENS6=/mnt/d/avv/ens/private_r6
OUT7=/mnt/d/avv/ens/private_r7
mkdir -p $FLD $OUT7
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
GPU=${1:-0}

PAIRS=""
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  M=/mnt/d/avv/output/${s}_gsplatB9ut
  [ -f $M/ckpt.pt ] || { echo "=== $s NO CKPT — ABORT ==="; exit 1; }

  # audit r9 CONFIRMED BUG: the field must be fit on the SAME pixel pipeline the
  # scene's test renders ship through. HNI0131/HNI0265 (k1=-0.115) ship via warp.
  UTR=native
  case $s in HNI0131|HNI0265) UTR=warp;; esac

  # audit r9 noise caveat: HCM1439 has only 103 train frames. D7's fit is stable at
  # ~120 pairs; a thin fit warps by the WRONG amount, which is worse than not warping.
  # Rendering is cheap, so buy pairs: stride 2 normally, every frame when frames are few.
  NTR=$(ls $PRI/$s/train/images | wc -l)
  ST=2; [ "$NTR" -lt 150 ] && ST=1
  echo "=== $s: $NTR train frames, stride $ST -> $((NTR/ST)) pairs, path=$UTR ==="

  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$GPU python gsplat_track/render_train.py \
    --ckpt $M/ckpt.pt --source $PRI/$s/train --images images \
    --out $M/train_renders_$UTR --stride $ST --ut_render $UTR \
    || { echo "=== $s TRAIN RENDER FAILED — ABORT ==="; exit 1; }

  conda activate fastgs2
  python gsplat_track/fit_field.py \
    --render_dir $M/train_renders_$UTR --gt_dir $PRI/$s/train/images \
    --out $FLD/${s}.npy \
    || { echo "=== $s FIT FAILED — ABORT ==="; exit 1; }

  # apply to the R6 ENSEMBLE renders (lossless PNG in, lossless PNG out).
  # --strict: the field must carry provenance proving it was fit on TRAIN photos.
  python gsplat_track/apply_field.py --strict \
    --in_dir $ENS6/$s/png --field $FLD/${s}.npy --out_dir $OUT7/$s/png \
    || { echo "=== $s APPLY FAILED — ABORT ==="; exit 1; }

  # count PNGs only: apply_field drops a field_applied.json stamp in out_dir (the
  # double-warp guard), which would otherwise read as an extra "image"
  n_in=$(ls $ENS6/$s/png/*.png 2>/dev/null | wc -l)
  n_out=$(ls $OUT7/$s/png/*.png 2>/dev/null | wc -l)
  [ "$n_in" = "$n_out" ] || { echo "=== $s COUNT MISMATCH $n_in vs $n_out — ABORT ==="; exit 1; }
  echo "=== PRIVFIELD $s DONE ($n_out imgs) ==="
  PAIRS="$PAIRS $s=$OUT7/$s/png"
done

echo "=== fitted field magnitudes (HNI0131/HNI0265 are the k1=-0.115 pair; a LARGE"
echo "===  field there would confirm the degenerate-fit story) ==="
python - <<'PY'
import numpy as np, glob, os
for f in sorted(glob.glob("/mnt/d/avv/fields/*.npy")):
    a = np.load(f); m = np.linalg.norm(a, axis=2)
    print(f"  {os.path.basename(f):14s} mean |d| {m.mean():.3f} px   max {m.max():.2f} px")
PY

python build_submission_zip.py --scene_dirs $PAIRS \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round7_field_private.zip \
  || { echo "=== R7 ZIP BUILD FAILED ==="; exit 1; }
echo "PRIVFIELD ZIP DONE"

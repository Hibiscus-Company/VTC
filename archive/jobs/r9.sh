#!/bin/bash
# R9 = 60k/8M UT members (replacing 30k/5M) + lens field REFIT from those members + mask.
#
# The members are the only thing left that moves the score: R8 (+0.002), R8b (-0.0001) and
# the old packaging sweep (-0.05 total) all agree the remaining gains are NOT in composition
# or packaging.
#
# Field is refit from the 60k model's own train renders (principled; the field is a camera
# property so reuse would likely do, but this is what the hardened pipeline does anyway).
# Mask kept: R8 proved it neutral (+0.002), but it fixes a real defect and now costs nothing
# thanks to the ladder reclaim.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
FLD=/mnt/d/avv/fields9
MSK=/mnt/d/avv/masks
R9=/mnt/d/avv/ens/private_r9
mkdir -p $FLD $R9
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1

SCENES="HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437"

# wait for every 60k member to be trained AND rendered.
# BOUNDED (audit r11): if a member train dies, an unbounded wait would hang forever
# instead of failing, and we would discover it hours later.
DEADLINE=$(( $(date +%s) + 12*3600 ))
while :; do
  miss=""
  for s in $SCENES; do
    d=/mnt/d/avv/output/${s}_gsplatB9ut60k
    [ -f $d/ckpt.pt ] && [ -d $d/test_poses_renders_png ] || miss="$miss $s"
  done
  [ -z "$miss" ] && break
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "=== R9 TIMED OUT after 12h still missing:$miss — ABORT ==="
    exit 1
  fi
  echo "waiting on:$miss"
  sleep 300
done
echo "=== all 8 60k members present ==="

PAIRS=""
for s in $SCENES; do
  M=/mnt/d/avv/output/${s}_gsplatB9ut60k
  UTR=native
  case $s in HNI0131|HNI0265) UTR=warp;; esac      # negative-k1: degenerate COLMAP fit
  NTR=$(ls $PRI/$s/train/images | wc -l)
  ST=2; [ "$NTR" -lt 150 ] && ST=1                 # a thin field fit warps by the WRONG amount

  # refit the field from THIS model's train renders, on the SAME path we ship
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_train.py \
    --ckpt $M/ckpt.pt --source $PRI/$s/train --images images \
    --out $M/train_png_$UTR --stride $ST --ut_render $UTR 2>&1 | tail -1 \
    || { echo "=== $s TRAIN RENDER FAILED — ABORT ==="; exit 1; }

  conda activate fastgs2
  python gsplat_track/fit_field.py --render_dir $M/train_png_$UTR \
    --gt_dir $PRI/$s/train/images --out $FLD/${s}.npy \
    || { echo "=== $s FIT FAILED — ABORT ==="; exit 1; }
  [ -f $MSK/${s}.npy ] || python gsplat_track/fov_mask.py \
    --sparse $PRI/$s/train/sparse/0 --out $MSK/${s}.npy || exit 1

  # 3 gates (masked: unsupervised in the outer ring on negative-k1 scenes) + 2 UT members
  python ensemble_renders.py \
    --dirs /mnt/d/avv/output/${s}_champA/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memB/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memC/test_poses_renders_png \
           $M/test_poses_renders_png \
           /mnt/d/avv/output/${s}_gsplatUTs7/test_poses_renders_png \
    --weights 0.1333 0.1333 0.1333 0.3 0.3 \
    --masks $MSK/${s}.npy $MSK/${s}.npy $MSK/${s}.npy none none \
    --out $R9/$s/jpg --png_dir $R9/$s/png_ens \
    --names_from $PRI/$s/test/test_poses.csv \
    || { echo "=== $s ENS FAILED — ABORT ==="; exit 1; }

  python gsplat_track/apply_field.py --strict \
    --in_dir $R9/$s/png_ens --field $FLD/${s}.npy --out_dir $R9/$s/png \
    || { echo "=== $s FIELD FAILED — ABORT ==="; exit 1; }

  n_in=$(ls $R9/$s/png_ens/*.png | wc -l); n_out=$(ls $R9/$s/png/*.png | wc -l)
  [ "$n_in" = "$n_out" ] || { echo "=== $s COUNT MISMATCH — ABORT ==="; exit 1; }
  echo "=== R9 $s DONE ($n_out imgs, $UTR, stride $ST) ==="
  PAIRS="$PAIRS $s=$R9/$s/png"
done

python build_submission_zip.py --scene_dirs $PAIRS \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round9_ut60k_private.zip \
  || { echo "=== R9 ZIP BUILD FAILED ==="; exit 1; }
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round9_ut60k_private.zip \
  --data_root $PRI || { echo "=== R9 VERIFY FAILED ==="; exit 1; }
echo "R9 ZIP DONE"

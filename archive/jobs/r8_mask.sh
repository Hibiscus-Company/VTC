#!/bin/bash
# R8 = R7 (lens field) + P6 masked composition on the two negative-k1 scenes.
#
# VALIDATED ON TRAIN GT (HNI0131, n=40): in the 12.9% ring the FastGS gates were never
# supervised on, they collapse to 17.80 dB while the UT member holds 21.13 dB (+3.34 dB).
# We were feeding that broken signal in at 0.4 weight. Mask the gates there; the UT
# members (supervised across the whole frame, in distorted space) carry the ring alone.
#
# Only HNI0131/HNI0265 change -- the mask is all-ones on every other scene, so the
# other 6 scenes are taken verbatim from R7.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
R7=/mnt/d/avv/ens/private_r7
R8=/mnt/d/avv/ens/private_r8
FLD=/mnt/d/avv/fields
MSK=/mnt/d/avv/masks
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
mkdir -p $R8

PAIRS=""
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0366 HNI0437; do
  PAIRS="$PAIRS $s=$R7/$s/png"          # unchanged from R7 (mask is a no-op there)
done

for s in HNI0131 HNI0265; do
  [ -f $MSK/${s}.npy ] || python gsplat_track/fov_mask.py \
      --sparse $PRI/$s/train/sparse/0 --out $MSK/${s}.npy || exit 1

  # 1) re-ensemble with the gates masked out of the unsupervised ring
  python ensemble_renders.py \
    --dirs /mnt/d/avv/output/${s}_champA/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memB/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memC/test_poses_renders_png \
           /mnt/d/avv/output/${s}_gsplatB9ut/test_poses_renders_png \
           /mnt/d/avv/output/${s}_gsplatUTs7/test_poses_renders_png \
    --weights 0.1333 0.1333 0.1333 0.3 0.3 \
    --masks $MSK/${s}.npy $MSK/${s}.npy $MSK/${s}.npy none none \
    --out $R8/$s/jpg --png_dir $R8/$s/png_ens \
    --names_from $PRI/$s/test/test_poses.csv \
    || { echo "=== R8 ens $s FAILED ==="; exit 1; }

  # 2) apply the SAME per-scene lens field (fit on that scene's train photos; it is a
  #    camera property and does not depend on ensemble composition)
  python gsplat_track/apply_field.py --strict \
    --in_dir $R8/$s/png_ens --field $FLD/${s}.npy --out_dir $R8/$s/png \
    || { echo "=== R8 field $s FAILED ==="; exit 1; }

  n_in=$(ls $R8/$s/png_ens/*.png | wc -l); n_out=$(ls $R8/$s/png/*.png | wc -l)
  [ "$n_in" = "$n_out" ] || { echo "=== $s COUNT MISMATCH $n_in vs $n_out — ABORT ==="; exit 1; }
  echo "=== R8 $s DONE (masked + field, $n_out imgs) ==="
  PAIRS="$PAIRS $s=$R8/$s/png"
done

python build_submission_zip.py --scene_dirs $PAIRS \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round8_maskfield_private.zip \
  || { echo "=== R8 ZIP BUILD FAILED ==="; exit 1; }
echo "R8 ZIP DONE"

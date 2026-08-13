#!/bin/bash
# FIELD CORRECTION: 5/5 public validation (standing rule -- validate on public
# with real test GT before rolling to private).
#
# For each public scene: render TRAIN views -> fit the displacement field on
# (train renders, TRAIN photos) -> apply it to that scene's TEST renders -> score.
# No test GT is ever used to fit. Expect ~+0.9 dB PSNR / ~+0.5 score pts.
#
# Runs on GPU0 ALONGSIDE its 60k trainer (6.5GB free there; a 5M-gaussian render
# needs ~2-3GB and an 8M one already succeeded there once). GPU1 is exp31 with
# 199MB free -- nothing can share it, so we do not wait for exp31.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PUB=~/data/phase1/public_set
FLD=/mnt/d/avv/fields
mkdir -p $FLD
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1

BEF=~/densq/fieldval_before
AFT=~/densq/fieldval_after
rm -rf $BEF $AFT; mkdir -p $BEF $AFT

for s in HCM0181 HCM0193 HCM0204 hcm0031 hcm0034; do
  OUT=/mnt/d/avv/output/${s}_gsplatB9ut
  [ -f $OUT/ckpt.pt ] || { echo "=== $s NO CKPT, skip ==="; continue; }

  conda activate gsplat
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_train.py \
    --ckpt $OUT/ckpt.pt --source $PUB/$s/train --images images \
    --out $OUT/train_renders --stride 4 \
    || { echo "=== $s TRAIN RENDER FAILED ==="; continue; }

  conda activate fastgs2
  python gsplat_track/fit_field.py \
    --render_dir $OUT/train_renders --gt_dir $PUB/$s/train/images \
    --out $FLD/${s}.npy \
    || { echo "=== $s FIT FAILED ==="; continue; }

  python gsplat_track/apply_field.py \
    --in_dir $OUT/test_poses_renders_png --field $FLD/${s}.npy \
    --out_dir $OUT/test_poses_renders_field \
    || { echo "=== $s APPLY FAILED ==="; continue; }

  ln -sfn $OUT/test_poses_renders_png $BEF/$s
  ln -sfn $OUT/test_poses_renders_field $AFT/$s
  echo "=== FIELD $s DONE ==="
done

conda activate fastgs2
echo "########## BEFORE (no field) ##########"
CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub $BEF --device cuda:0 2>&1 | grep -E "^[A-Za-z]+[0-9]+ |Score"
echo "########## AFTER (train-fitted field) ##########"
CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub $AFT --device cuda:0 2>&1 | grep -E "^[A-Za-z]+[0-9]+ |Score"
echo "FIELDVAL DONE"

#!/bin/bash
# Build the full 13-scene submission with the champion config:
#   g15 (grad_abs 0.00015) + --lambda_lpips 0.1 @25k + --metric_gate 1
# Public scenes reuse the already-trained/validated exp10/exp12 models;
# private-8 are trained here. Zip goes to /mnt/d/avv/submissions/.
#
# Usage: bash build_submission_champ.sh <tag>   (e.g. round1_champ)
set -o pipefail
cd "$(dirname "$0")"
TAG=${1:?usage: build_submission_champ.sh <tag>}
PUB=~/data/phase1/public_set
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2

# 1) train + render private-8 (undistort first if missing)
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  if [ ! -d "$PRI/$s/train/images_undist" ]; then
    python undistort_scene.py "$PRI/$s/train" || exit 1
  fi
done
MULT=0.7 DENSIFY_INT=100 HIGHFEAT_LR=0.04 GRAD_ABS=0.00015 \
IMAGES_DIR=images_undist DISTORT=auto \
EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate 1" \
bash run_scenes.sh 1 champ \
  $PRI/HCM0249 $PRI/HCM0254 $PRI/HCM0276 $PRI/HCM1439 \
  $PRI/HNI0131 $PRI/HNI0265 $PRI/HNI0366 $PRI/HNI0437 \
  2>&1 | grep -vE "Training progress|it/s" | grep -E "===|FAILED|Error"

# 2) assemble: public scenes use each scene's BEST-scoring renders (local GT
#    validation, per-scene selection), private from champ models
STAGE=~/subs_eval/sub_$TAG
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -r output/HCM0181_e10gate1/test_poses_renders "$STAGE/HCM0181"   # .7515 champ
cp -r output/hcm0031_e06stack/test_poses_renders "$STAGE/hcm0031"   # .7373 > champ .7369
for s in HCM0193 HCM0204 hcm0034; do
  cp -r output/${s}_e12champ/test_poses_renders "$STAGE/$s"
done
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  cp -r output/${s}_champ/test_poses_renders "$STAGE/$s"
done

# 3) validate names/count/sizes against every CSV + zip
python make_submission.py --renders_root "$STAGE" --out /mnt/d/avv/submissions/sub_${TAG}.zip \
  --data_roots "$PUB" "$PRI" || exit 1
echo "SUBMISSION DONE: /mnt/d/avv/submissions/sub_${TAG}.zip"

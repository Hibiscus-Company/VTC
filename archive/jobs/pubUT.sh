#!/bin/bash
# exp27 follow-up on GPU1 (idle after trackb7):
#  A. HCM0181 ensemble A/B with the new best single B9ut (75.2143):
#     m4 (gate1+2+3+0, round-3 baseline 75.980) vs +B9ut vs +B8pure vs both.
#     Round-7 rule: different-renderer members help ADDITIVELY only.
#  B. Idea-2 public validation: identical exp27 config on the other 4 public
#     scenes (user rule: validate on public before rolling to private).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PUB=~/data/phase1/public_set
CSV=$PUB/HCM0181/test/test_poses.csv
ENS=/mnt/d/avv/ens/hcm0181_ab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2

G1=output/HCM0181_e10gate1/test_poses_renders
G2=output/HCM0181_e08gate2/test_poses_renders
G3=output/HCM0181_e11gate3/test_poses_renders
G0=output/HCM0181_e13gate0/test_poses_renders_png
UT=/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png
PURE=/mnt/d/avv/output/HCM0181_gsplatB8pure/test_poses_renders_png

ens_ab() {  # name, member dirs...
  local NAME=$1; shift
  python ensemble_renders.py --dirs "$@" --out $ENS/$NAME --names_from $CSV \
    || { echo "=== ens $NAME FAILED ==="; return 1; }
  mkdir -p ~/densq/ab_$NAME
  ln -sfn $ENS/$NAME ~/densq/ab_$NAME/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/ab_$NAME --device cuda:0 2>&1 \
    | grep -E "Score\(vgg\)" | sed "s/^/AB $NAME /"
}
ens_ab m4        $G1 $G2 $G3 $G0
ens_ab m5ut      $G1 $G2 $G3 $G0 $UT
ens_ab m5pure    $G1 $G2 $G3 $G0 $PURE
ens_ab m6utpure  $G1 $G2 $G3 $G0 $UT $PURE
echo "ENSEMBLE AB DONE"

for s in hcm0031 hcm0034 HCM0193 HCM0204; do
  NAME=${s}_gsplatB9ut
  OUT=/mnt/d/avv/output/$NAME
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
    --source $PUB/$s/train --images images --ut \
    --out $OUT --iters 30000 --cap_max 5000000 --noise_stop 25000 \
    || { echo "=== $NAME TRAIN FAILED ==="; continue; }
  echo "=== $NAME TRAIN DONE ==="
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $PUB/$s/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  mkdir -p ~/densq/utpub_$s
  ln -sfn $OUT/test_poses_renders ~/densq/utpub_$s/$s
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/utpub_$s --device cuda:0 2>&1 \
    | grep -E "$s |Score\(vgg\)"
  echo "=== $NAME SCORED ==="
done
echo "PUBUT QUEUE DONE"

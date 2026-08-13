#!/bin/bash
# exp30 = UT seed-jitter ensemble member: exp28 config (cap8M) with --seed 7.
# Different data order under MCMC relocation/noise => decorrelated model in
# the SAME (strongest) family. Ensemble A/B vs mixed-family composition:
#   m5utA  = 4 gates + UT8M(seed42)          [current private composition +8M]
#   utpair = UT8M(seed42) + UT8M(seed7) only [2-member, same family]
#   m6ut2  = 4 gates + both UT8M seeds
# Decides round-6 composition. Chains after exp29 on GPU1; keeps GPU1 hot.
set -o pipefail
until grep -q "TRACKB9 QUEUE DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/trackb9.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
CSV=$S/test/test_poses.csv
source ~/miniconda3/etc/profile.d/conda.sh
NAME=gsplatB12ut8Ms7
OUT=/mnt/d/avv/output/HCM0181_$NAME

conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $S/train --images images --ut --seed 7 \
  --out $OUT --iters 30000 --cap_max 8000000 --noise_stop 25000 \
  || { echo "=== $NAME TRAIN FAILED ==="; exit 1; }
echo "=== $NAME TRAIN DONE ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/ckpt.pt --csv $CSV \
  --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
  --ut_render native 2>&1 | tail -1
conda activate fastgs2
mkdir -p ~/densq/$NAME
ln -sfn $OUT/test_poses_renders ~/densq/$NAME/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$NAME --device cuda:0 2>&1 | grep -E "HCM0181 |Score\(" | sed 's/^/S7 single /'

G1=output/HCM0181_e10gate1/test_poses_renders
G2=output/HCM0181_e08gate2/test_poses_renders
G3=output/HCM0181_e11gate3/test_poses_renders
G0=output/HCM0181_e13gate0/test_poses_renders_png
UT42=/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png
UT7=$OUT/test_poses_renders_png
ENS=/mnt/d/avv/ens/hcm0181_ab2

ens_ab() {
  local NAME2=$1; shift
  python ensemble_renders.py --dirs "$@" --out $ENS/$NAME2 --names_from $CSV \
    || { echo "=== ens $NAME2 FAILED ==="; return 1; }
  mkdir -p ~/densq/ab2_$NAME2
  ln -sfn $ENS/$NAME2 ~/densq/ab2_$NAME2/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/ab2_$NAME2 --device cuda:0 2>&1 \
    | grep -E "Score\(vgg\)" | sed "s/^/AB2 $NAME2 /"
}
ens_ab m5utA   $G1 $G2 $G3 $G0 $UT42
ens_ab utpair  $UT42 $UT7
ens_ab m6ut2   $G1 $G2 $G3 $G0 $UT42 $UT7
echo "TRACKB10 QUEUE DONE"

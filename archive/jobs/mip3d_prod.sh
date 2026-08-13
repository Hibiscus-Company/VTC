#!/bin/bash
# r25 long pole: a mip3d member for each of the 5 set2 towers, trained on the FULL train set.
#
# VALIDATED ON TWO SCENES, paired, all three metrics up both times:
#   HCM0181 (180/240 data)  74.9974 -> 75.6570   +0.6596
#   HCM0421 (200/240 data)  75.7616 -> 76.2188   +0.4572
# HONEST CAVEAT: the gain shrinks as training data grows (+8pp data cost -0.20), the same
# starvation trend that is sinking the restoration head -- mip3d is a REGULARIZER, and regularizers
# help most when data is scarce. Production has 240 images, so expect less than +0.46.
# THAT IS WHY THESE ARE ADDED AS EXTRA ENSEMBLE MEMBERS, NOT SWAPPED IN: adding a decorrelated
# member is a COMPOSITION change, the one class that has never transferred below 1x in 10 graded
# rounds. If mip3d's quality edge survives at 240 images we get that too; if it evaporates we still
# hold a normal decorrelated member and cannot go backwards.
# New seed 555 (existing members are seeds 7/42/13/77/101) to maximise decorrelation.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
OUT=/mnt/d/avv/r25_mip3d
CLAIMS=$OUT/claims
mkdir -p "$CLAIMS"
TOWERS="HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"

gpu_free() {
  local m; m=$(nvidia-smi --id=$1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-99999}" -lt 2000 ]
}

run_one() {
  local G=$1 T=$2 M=$OUT/$2
  conda activate gsplat
  echo ">>> gpu$G $T mip3d train $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $DATA/$T/train --images images \
    --seed 555 --ut --ema_decay 0.999 --mip3d 0.2 --mip3d_every 100 --out $M \
    --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 \
    | grep -aE "Baked|Saved" | tail -2 || { echo "!!! $T TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $DATA/$T/test/test_poses.csv --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 \
    | tail -1 || { echo "!!! $T RENDER FAIL"; return 1; }
  rm -f $M/ckpt.pt
  echo "<<< gpu$G $T done $(date +%H:%M) ($(ls $M/test_png | wc -l) pngs)"
  touch $OUT/${T}.DONE
}

worker() {
  local G=$1
  for T in $TOWERS; do
    if mkdir "$CLAIMS/$T" 2>/dev/null; then
      # serialise on a lock so two workers can never race into the same free GPU
      # (the 05:55 contention that cost 5h of GPU0 today)
      while ! gpu_free "$G"; do sleep 60; done
      run_one "$G" "$T"
    fi
  done
}

worker 0 &
sleep 90          # stagger so the two workers never evaluate "free" on the same instant
worker 1 &
wait
echo "=== MIP3D PRODUCTION MEMBERS DONE $(date) ==="
touch /mnt/d/avv/mip3d_prod.DONE

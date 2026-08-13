#!/bin/bash
# Productionize the VALIDATED seed-101 decorrelated member across the 5 set2 towers -> native test
# renders = safe reusable r22-prep artifacts (field/ensemble/build deferred to a verified step; NEVER
# auto-submit). Base seed (no ema_decay) = max decorrelation from the existing ut7_ema999 members;
# eval-split proved it adds +0.64 as a 1->2 member and renders correctly aligned in native mode.
# Claim-based round-robin: GPU1 starts now; GPU0 joins after seed202_bank.DONE (no stacking on the
# running seed-202). Each tower: train full data -> render test_poses native -> DELETE ckpt (disk).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
OUT=/mnt/d/avv/r22_seed101
CLAIMS=$OUT/claims
mkdir -p "$CLAIMS"
TOWERS="HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"
RECIPE="--ut --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

run_one() {  # $1 gpu  $2 tower
  local G=$1 T=$2 M=$OUT/$2
  conda activate gsplat
  echo ">>> gpu$G $T train $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $DATA/$T/train --images images \
    --seed 101 $RECIPE --out $M 2>&1 | tail -2 || { echo "!!! $T TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $DATA/$T/test/test_poses.csv --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 \
    | tail -1 || { echo "!!! $T RENDER FAIL"; return 1; }
  rm -f $M/ckpt.pt   # disk hygiene: production member retrains here anyway; keep only renders
  echo "<<< gpu$G $T done $(date +%H:%M) ($(ls $M/test_png | wc -l) test pngs)"
  touch $OUT/${T}.DONE
}

worker() {  # $1 gpu
  local G=$1
  for T in $TOWERS; do
    if mkdir "$CLAIMS/$T" 2>/dev/null; then run_one "$G" "$T"; fi
  done
}

worker 1 &                                        # GPU1 free now
( while [ ! -f /mnt/d/avv/seed202_bank.DONE ]; do sleep 60; done; worker 0 ) &   # GPU0 after seed-202
wait
echo "=== SEED101 PROD POOL DONE $(date) ==="
touch /mnt/d/avv/seed101_prod_pool.DONE

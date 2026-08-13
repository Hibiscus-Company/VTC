#!/bin/bash
# r28 member bank: mip3d members where we have none, then a SECOND mip3d member per tower.
#
# WHY THIS AND NOT ANOTHER POST-PROCESSING IDEA: the post-processing axis is now exhausted. This
# session alone closed the resampling kernel (r27, shipped, +0.0743), the band-limited warp, the
# field estimator, per-image exposure, ensemble member alignment, and the chair field; the campaign
# had already killed sharpening, grain, SSAA, colour correction, texture injection and within-family
# weights. COMPOSITION is the one lever that has never transferred below 1x in 12 graded rounds,
# and mip3d is our best member family (+0.6596 solo on HCM0181, +0.4572 on HCM0421, all three
# metrics up both times).
#
# JOB ORDER IS BY EXPECTED VALUE PER GPU-HOUR, because we will not finish all seven:
#  1-2. chair and bonsai have NO mip3d member at all, so these are NEW-FAMILY adds (the r25 tower
#       precedent for a new-family add was +0.0596/scene on the leaderboard). They are also the only
#       scenes that train WITHOUT --ut, i.e. with gsplat's antialiased rasteriser, so mip3d there is
#       the 3D filter ON TOP OF the 2D screen-space filter = full Mip-Splatting, which we have never
#       actually run. bonsai is only 30k/5M so it costs ~1.2h.
#  3-7. a SECOND mip3d member per tower (seed 777). Same-family, so no new-family bonus, but it
#       lifts the mip3d family weight from 0.20 to ~0.33 of the ensemble.
# Each finished member is independently usable: build r28 from whatever has landed at the cutoff.
#
# GPU DISCIPLINE (this has cost us 5h before): claim-then-wait, and wait for a GENUINELY free GPU
# (<2000 MiB) so we never stack on top of the verification jobs. 90s stagger so two workers cannot
# evaluate "free" on the same instant.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
PRI=/mnt/d/avv/data/phase1/private_set2
OUT=/mnt/d/avv/r28_members
CLAIMS=$OUT/claims
mkdir -p "$CLAIMS"

JOBS="chair bonsai HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"

args_for() {
  case "$1" in
    chair)  echo "--seed 555 --ema_decay 0.99 --mip3d 0.2 --mip3d_every 100 --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000" ;;
    bonsai) echo "--seed 555 --mip3d 0.2 --mip3d_every 100 --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000" ;;
    *)      echo "--seed 777 --ut --ema_decay 0.999 --mip3d 0.2 --mip3d_every 100 --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000" ;;
  esac
}

gpu_free() {
  local m; m=$(nvidia-smi --id=$1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-99999}" -lt 2000 ]
}

run_one() {
  local G=$1 T=$2 M=$OUT/$2
  local A; A=$(args_for "$T")
  local UTR=""
  case "$T" in chair|bonsai) UTR="" ;; *) UTR="--ut_render native" ;; esac
  conda activate gsplat
  echo ">>> gpu$G $T START $(date +%H:%M)  [$A]"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $PRI/$T/train --images images --out $M $A 2>&1 \
    | grep -aE "Baked|Saved|Error|error" | tail -3 || { echo "!!! $T TRAIN FAIL"; return 1; }
  [ -f $M/ckpt.pt ] || { echo "!!! $T NO CKPT"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $PRI/$T/test/test_poses.csv --out $M/test_render --png_dir $M/test_png $UTR 2>&1 \
    | tail -1 || { echo "!!! $T RENDER FAIL"; return 1; }
  echo "$A" > $M/train_args.txt
  # the ckpt is ~2GB at 8M gaussians and disk is at 29GB free, so it is reclaimed OUTSIDE this
  # script by name once the renders are on disk -- see the DONE flag below
  echo "<<< gpu$G $T DONE $(date +%H:%M)  ($(ls $M/test_png 2>/dev/null | wc -l) pngs)"
  touch $OUT/${T}.DONE
}

worker() {
  local G=$1
  for T in $JOBS; do
    if mkdir "$CLAIMS/$T" 2>/dev/null; then
      while ! gpu_free "$G"; do sleep 60; done
      run_one "$G" "$T" || echo "!!! $T FAILED on gpu$G, continuing"
    fi
  done
  echo "=== worker gpu$G drained $(date +%H:%M) ==="
}

worker 0 &
sleep 90
worker 1 &
wait
echo "=== R28 MEMBER BANK COMPLETE $(date) ==="
touch /mnt/d/avv/r28_members.DONE

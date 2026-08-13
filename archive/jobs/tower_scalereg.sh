#!/bin/bash
# THE HIGHEST-LEVERAGE UNTESTED EXTENSION: does scale_reg transfer from bonsai to the TOWERS?
#
# WHY it matters more than anything else on the board: scale_reg=0.1 is worth +0.139 on the bonsai
# SCENE (LB-confirmed at r35). Towers are FIVE of the seven scenes. If it transfers at the same
# rate, that is 5 x 0.139 / 7 = +0.099 LB -- five times the r35 win. Even at one-third strength it
# beats every other item in the ledger.
#
# WHY it might not transfer, stated before looking:
#   1. bonsai is capacity-starved (84.4% of its held-out LPIPS is an in-sample fitting floor).
#      Towers are not: 240 train views, 8M cap, 60k iters, and they already sit at 78.74.
#      The mechanism that made larger gaussians pay on bonsai may simply be absent.
#   2. REGIME MISMATCH. bonsai runs 30k iters; towers run 60k. Two readings of what to port:
#        - budget-matched: effect ~ total reg gradient applied -> lam_tower = 0.05
#        - balance-matched: at convergence the data/reg balance is set by lam alone -> lam = 0.1
#      Both are defensible, and on bonsai lam=0.3 was already PAST the peak (71.7553, below bar),
#      so guessing wrong in the high direction actively costs. Hence: measure the curve, do not
#      pick one and hope.
#
# DESIGN: 3 single arms (0.01 control, 0.05, 0.1) as a SCREEN for curve shape -- exactly how the
# bonsai curve was found (6 single arms, clean unimodal, peak at 0.1). A single 1-vs-1 A/B sits
# under the 0.407 run-to-run noise floor, so NO shipping decision comes out of this run; if a peak
# appears, it gets the diversity-matched k=2 paired gate before anything is built.
#
# Recipe is the SHIPPED tower recipe verbatim (hcm0421_members.sh) -- --ut, 60k, 8M, ema 0.999,
# native UT render. scale_reg is the ONLY variable. Model-class mismatch has burned this campaign
# before (a claimed +0.61 was really +0.192 because the arms were --ut and the baseline was not).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/HCM0421; OUT=/mnt/d/avv/r41_towersr; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
mkdir -p $OUT
BASE="--ut --ema_decay 0.999 --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 --seed 42"
arm(){ local TAG=$1 SR=$2 M=$OUT/$1
  [ -f $OUT/$TAG.DONE ] && { echo "$TAG done, skip"; return 0; }
  # take whichever GPU is free first; never stack (a stacked run already cost us once)
  local G=""
  while [ -z "$G" ]; do
    for g in 0 1; do
      [ "$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits|tr -d ' ')" -lt 3000 ] && { G=$g; break; }
    done
    [ -z "$G" ] && sleep 60
  done
  mkdir -p $M; conda activate gsplat
  echo ">>> $TAG scale_reg=$SR (gpu$G) START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --out $M $BASE --scale_reg $SR 2>&1 | grep -aE "Saved|Error|error" | tail -2
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT"; touch $OUT/$TAG.DONE; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  python $T/census.py $M/ckpt.pt
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag $TAG
  rm -f $M/ckpt.pt
  echo "<<< $TAG DONE $(date +%H:%M)"; touch $OUT/$TAG.DONE; }
arm sr001 0.01
arm sr005 0.05
arm sr01  0.1
echo "=== TOWER SCALEREG SCREEN DONE $(date +%H:%M) ==="; touch $OUT/ALL.DONE

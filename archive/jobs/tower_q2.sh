#!/bin/bash
# Tower scale_reg screen -- GPU1. RESTART: the 09:30 launch died with the parent process at ~11:37
# having burned 2h07 with no output. Same three arms, same shipped tower recipe, detached properly.
#
# The bet: scale_reg=0.1 is worth +0.139 on the bonsai SCENE (LB-confirmed at r35). Towers are FIVE
# of the seven scenes; the same rate would be 5 x 0.139 / 7 = +0.099 LB.
# Two pre-registered reasons it may not transfer: towers are not capacity-starved, and they run 60k
# iters vs bonsai's 30k (budget-matching says port 0.05, balance-matching says 0.1, and 0.3 was
# already past the peak on bonsai). Hence a 3-point SCREEN, not a guessed value.
# Single arms sit under the 0.407 training-noise floor -> NO ship decision comes out of this; a peak
# earns the diversity-matched encoded k=2 gate first.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/HCM0421; OUT=/mnt/d/avv/r41_towersr; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
mkdir -p $OUT
BASE="--ut --ema_decay 0.999 --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 --seed 42"
G=1
arm(){ local TAG=$1 SR=$2; local M=$OUT/$TAG
  [ -f $OUT/$TAG.DONE ] && { echo "$TAG done, skip"; return 0; }
  mkdir -p $M; conda activate gsplat
  echo ">>> $TAG scale_reg=$SR (gpu$G) START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --out $M $BASE --scale_reg $SR > $M/train.log 2>&1
  grep -aE "^Saved" $M/train.log | tail -1
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT"; tail -3 $M/train.log; touch $OUT/$TAG.DONE; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native > $M/render.log 2>&1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python $T/census.py $M/ckpt.pt
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag $TAG
  rm -f $M/ckpt.pt
  echo "<<< $TAG DONE $(date +%H:%M)"; touch $OUT/$TAG.DONE; }
arm sr001 0.01
arm sr005 0.05
arm sr01  0.1
echo "=== TOWER SCREEN DONE $(date +%H:%M) ==="; touch $OUT/ALL.DONE

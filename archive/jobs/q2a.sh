#!/bin/bash
# GPU0 follow-up. Waits for minop002 to release the card (~22:15).
#
# ARM 1 -- churn_r25n25_s101. THE GATE ARM, and the reason it goes first: churn_r25n25 (+0.302)
# is a measurement, not something shippable. The pre-registered gate for a training-recipe change
# is the diversity-matched encoded k=2 paired test, which needs two members per side. The
# reference side already has both (r36_shape/sr01 seed 42, r38/sr01_s101 seed 101); the churn side
# has only seed 42. This one run completes the pair and is the ONLY thing standing between the
# result and a submission decision. 1h35.
#
# ARM 2 -- long60k. The record kills 60k for bonsai (70.72 vs 30k's 71.16), but that arm ran under
# the starved churn schedule. 25k/30k is 83% of the run and the shipped tower recipe uses
# 50k/60k = the same 83%, so churn_r25n25 is already the tower ratio at 30k and bonsai at 60k with
# the tower's schedule has never been run. If longer training was only losing because topology
# froze at 25k, this recovers it. ~3h10.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai; OUT=/mnt/d/avv/r43_bonsai; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
COMMON="--cap_max 5000000 --scale_reg 0.1"
G=0
while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits|tr -d ' ')" -gt 3000 ]; do sleep 120; done
arm(){ local TAG=$1; shift; local M=$OUT/$TAG
  [ -f $OUT/$TAG.DONE ] && { echo "$TAG done, skip"; return 0; }
  mkdir -p $M; conda activate gsplat
  echo ">>> $TAG [$*] (gpu$G) START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --out $M $COMMON "$@" > $M/train.log 2>&1
  grep -aE "^Saved" $M/train.log | tail -1
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT"; tail -3 $M/train.log; touch $OUT/$TAG.DONE; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png > $M/render.log 2>&1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python $T/census.py $M/ckpt.pt
  CUDA_VISIBLE_DEVICES=$G python $T/score_split.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag $TAG 2>&1 | grep -avE "Warning|warn|torchvision|Loading model|Setting up"
  echo "$COMMON $*" > $M/train_args.txt
  rm -f $M/ckpt.pt
  echo "<<< $TAG DONE $(date +%H:%M)"; touch $OUT/$TAG.DONE; }

arm churn_r25n25_s101 --iters 30000 --refine_stop 25000 --noise_stop 25000 --lpips_from 12000 --seed 101
arm long60k           --iters 60000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 --seed 42
echo "=== Q2A DONE $(date +%H:%M) ==="; touch $OUT/Q2A.DONE

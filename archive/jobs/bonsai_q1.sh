#!/bin/bash
# bonsai arm queue #1 -- GPU0. Six single-variable arms on top of the LB-confirmed scale_reg=0.1.
#
# WHY THESE SIX, after this morning's evidence:
#  * A4 mined the record and found the churn knobs (refine_stop / noise_stop) have exactly TWO
#    points on them: capD's 15k/8k, chosen mid-incident on 17/07 and never revisited in 13 days,
#    and "standard" (= the fog collapse). That is the same class of untouched defensive default
#    that just paid +0.139 LB via scale_reg. Arms 1-2.
#  * The census says only ~1.3M of the 5,000,000 saved gaussians have opacity > 0.05 -- 74% of the
#    budget is invisible -- and opacity_reg has NEVER been swept on bonsai. Arms 3-4 sweep the
#    coefficient; arms 5-6 attack the same waste from the relocation side (min_opacity is the
#    MCMC relocation threshold; raising it recycles dead gaussians into useful ones).
#    NOTE chair_minop02 COLLAPSED at 0.02 (26.99), so arm 6 is the deliberate risk of the batch.
#  * Capacity is NOT in this batch: 16 GB cards, and A4 re-based the 8M kill to -1.713 vs the
#    corrected bar (not the -0.7101 the record cited). That axis is closed harder than we thought.
#
# REFERENCE is sr01 = 71.9911 (scale_reg=0.1, seed 42, same split) -- a matched paired baseline,
# NOT the 71.799 default-scale_reg bar. Same seed, same everything but the one variable.
#
# Every arm reports first8 / last20 SEPARATELY (A3: the 28 holes are two populations, 62.00 vs
# 75.99, and a scene-averaged number hides a 14-point gap).
#
# DETACHED with setsid: the previous queue died when the parent Claude Code process exited,
# costing 2h07 of GPU0 and 10h30 of GPU1 with nothing to show. Not again.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai; OUT=/mnt/d/avv/r43_bonsai; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
mkdir -p $OUT
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 --scale_reg 0.1 --seed 42"
G=0
arm(){ local TAG=$1; shift; local M=$OUT/$TAG
  [ -f $OUT/$TAG.DONE ] && { echo "$TAG done, skip"; return 0; }
  mkdir -p $M; conda activate gsplat
  echo ">>> $TAG [$*] (gpu$G) START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --out $M $BASE "$@" > $M/train.log 2>&1
  grep -aE "^Saved" $M/train.log | tail -1
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT -- see $M/train.log"; tail -3 $M/train.log; touch $OUT/$TAG.DONE; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png > $M/render.log 2>&1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python $T/census.py $M/ckpt.pt
  CUDA_VISIBLE_DEVICES=$G python $T/score_split.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag $TAG
  echo "$BASE $*" > $M/train_args.txt
  rm -f $M/ckpt.pt
  echo "<<< $TAG DONE $(date +%H:%M)"; touch $OUT/$TAG.DONE; }

arm churn_r25n15 --refine_stop 25000 --noise_stop 15000
arm churn_r25n25 --refine_stop 25000 --noise_stop 25000
arm opreg003     --opacity_reg 0.003
arm opreg03      --opacity_reg 0.03
arm minop001     --min_opacity 0.01
arm minop002     --min_opacity 0.02
echo "=== BONSAI Q1 DONE $(date +%H:%M) ==="; touch $OUT/Q1.DONE

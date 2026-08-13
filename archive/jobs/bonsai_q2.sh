#!/bin/bash
# bonsai queue #2 -- GPU1, starts when the tower screen releases it (~20:05).
#
# WHY: q1 found the churn window is LIVE and the gain lands exactly where the diagnosis said it
# would. refine_stop/noise_stop 15k/8k -> 25k/15k -> 25k/25k gives +0.136 -> +0.302, monotone,
# and the effective gaussian count (opacity > 0.05) went 1.30M -> 1.95M, i.e. 50% more of the 5M
# budget became visible. Decomposed by population:
#     churn_r25n25   overall +0.302   first8 +1.265   last20 -0.084
# 8/28 x 1.265 + 20/28 x (-0.084) = +0.301 -- the ENTIRE gain is in the sparse-coverage holes,
# the population that partial correlation confirmed at p=0.003. Mechanism and diagnosis agree.
#
# ARM 1 -- long60k. THE BIG ONE. The record kills 60k iters (70.72 vs 30k's 71.16), but that arm
# ran under the STARVED churn schedule. 25k/30k is 83% of the run; the shipped tower recipe uses
# 50k/60k = 83% of the run. So r25n25 is already the tower ratio at 30k, and nobody has ever
# tested bonsai at 60k with the tower's churn schedule -- the "60k is worse" result is confounded
# with the churn defect we just fixed. If longer training was only losing because the topology
# froze at 25k, this recovers it. ~3h10.
#
# ARM 2 -- churn_r29n29. Pushes the confirmed axis PAST the tower ratio to the boundary
# (iters=30000). Tells us whether 25k/25k is a peak or just the far end of what was sampled. ~1h35.
#
# ARM 3 -- longer at the sweet spot if arm 1 wins; a 45k midpoint to locate the optimum. ~2h20.
#
# Reference throughout is sr01 = 71.9911 with first8 62.0041 / last20 75.9861.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai; OUT=/mnt/d/avv/r43_bonsai; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
mkdir -p $OUT
COMMON="--cap_max 5000000 --scale_reg 0.1 --seed 42"
G=1
# wait for the tower screen to release GPU1
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
  CUDA_VISIBLE_DEVICES=$G python $T/score_split.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag $TAG 2>&1 | grep -vE "Warning|warn|torchvision|Loading model|Setting up"
  echo "$COMMON $*" > $M/train_args.txt
  rm -f $M/ckpt.pt
  echo "<<< $TAG DONE $(date +%H:%M)"; touch $OUT/$TAG.DONE; }

arm long60k     --iters 60000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000
arm churn_r29n29 --iters 30000 --refine_stop 29000 --noise_stop 29000 --lpips_from 12000
arm long45k     --iters 45000 --refine_stop 38000 --noise_stop 38000 --lpips_from 30000
echo "=== BONSAI Q2 DONE $(date +%H:%M) ==="; touch $OUT/Q2.DONE

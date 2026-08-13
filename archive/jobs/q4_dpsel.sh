#!/bin/bash
# SELECTIVE depth prior -- the direct consequence of 31/07's measurement that a UNIFORM depth
# prior is a pure dial between the two hole populations:
#     dp     first8    last20
#     0      64.7180   75.8967
#     0.002  65.2131   75.3924
#     0.01   66.0625   74.7697
#     0.05   66.3579   72.1889     <- best first8 ever measured (+4.35 vs ref)
# Every column monotone, no interior optimum. The sparse holes want the prior; the dense ones are
# destroyed by it. So redistribute the SAME total budget by training-view coverage sparsity.
#
# train_gsplat_cov.py adds --depth_cov_pow P: per-view weight ~ (d5_i/median(d5))^P, normalised to
# MEAN 1, so P changes only the distribution and never the total. P=0 is bit-identical to uniform.
# On bonsai train_sub (220 views, d5 med 0.4326, max/med 4.0x):
#     P=1   sparsest 20% of views get 45.3% of the budget   (uniform = 20%)
#     P=2   sparsest 20% get 64.0%, w range 0.007..3.87
# P>=3 is LESS concentrated (61.8%, 54.6%) because it hits the clip at 8.
#
# Base is long45k (72.7028, gate-passed at +0.9043 paired), so each arm is a clean single-variable
# A/B. PRE-REGISTERED success: first8 > 65.5 AND last20 > 75.5 simultaneously -- i.e. it must keep
# most of dp05's first8 gain WITHOUT paying dp05's last20 price. Either alone is not a pass.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai; OUT=/mnt/d/avv/r43_bonsai; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
COMMON="--cap_max 5000000 --scale_reg 0.1 --iters 45000 --refine_stop 38000 --noise_stop 38000 --lpips_from 30000 --seed 42"
DP="--depth_dir /mnt/d/avv/depth/bonsai_sub"
G=$1; shift
# WAIT ON THE PRODUCING QUEUE'S OWN MARKER, never on GPU memory. On 31/07 a memory poll raced the
# production queue's 1-minute member transition: both queues saw the card free at 19:17, stacked
# two trainings on it, and production fell to ~23% of its normal step rate for 1h41.
echo "waiting for /mnt/d/avv/r45_prod/G$G.DONE before touching gpu$G ..."
while [ ! -f /mnt/d/avv/r45_prod/G$G.DONE ]; do sleep 60; done
# belt and braces: the card must also actually be free, twice, 3 minutes apart
for _ in 1 2; do
  while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits|tr -d ' ')" -gt 3000 ]; do sleep 60; done
  sleep 180
done
echo "gpu$G confirmed free $(date +%H:%M)"
arm(){ local TAG=$1; shift; local M=$OUT/$TAG
  [ -f $OUT/$TAG.DONE ] && { echo "$TAG done, skip"; return 0; }
  mkdir -p $M; conda activate gsplat
  echo ">>> $TAG [$*] (gpu$G) START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat_cov.py --source $ES/train_sub --images images \
    --out $M $COMMON "$@" > $M/train.log 2>&1
  grep -aE "^Saved|^depth prior redistributed" $M/train.log | tail -2
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT"; tail -5 $M/train.log; touch $OUT/$TAG.DONE; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png > $M/render.log 2>&1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python $T/census.py $M/ckpt.pt
  CUDA_VISIBLE_DEVICES=$G python $T/score_split.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag $TAG 2>&1 | grep -avE "Warning|warn|torchvision|Loading model|Setting up"
  echo "$COMMON $*" > $M/train_args.txt
  # KEEP THE WEIGHTS. The organiser now requires the winning checkpoints to be submitted,
  # and this exact `rm` already destroyed 50 of r36's 62 checkpoints. Archive first, then
  # drop the working copy so the training dir stays small.
  mkdir -p /mnt/d/avv/WEIGHTS
  cp -f $M/ckpt.pt /mnt/d/avv/WEIGHTS/bonsai_$TAG.pt && echo "  archived -> WEIGHTS/bonsai_$TAG.pt"
  rm -f $M/ckpt.pt
  echo "<<< $TAG DONE $(date +%H:%M)"; touch $OUT/$TAG.DONE; }
for spec in "$@"; do arm ${spec%%|*} ${spec#*|}; done
echo "=== DPSEL gpu$G DONE $(date +%H:%M) ==="; touch $OUT/DPSEL$G.DONE

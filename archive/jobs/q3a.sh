#!/bin/bash
# GPU0. long45k is the new best recipe (+0.7117) -- these two arms make it shippable and locate
# the length optimum.
#
# ARM 1 -- long45k_s101. THE GATE ARM for the new best. long45k has one seed; the pre-registered
# encoded k=2 paired gate needs two per side. Without this, +0.7117 stays a measurement.
#
# ARM 2 -- long50k. The length curve is a TRADE-OFF between the two hole populations and 45k sits
# near its top:
#     30k  first8 +1.27  last20 -0.08  ->  net +0.302
#     45k  first8 +2.71  last20 -0.09  ->  net +0.712
#     60k  first8 +3.04  last20 -0.86  ->  net +0.257
# first8 improves monotonically with length (the sparse-coverage holes need more optimisation to
# resolve under-constrained geometry) while last20 collapses past 45k (the well-covered holes
# start over-fitting). 50k tells us whether the peak is at 45k or between 45k and 60k.
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
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT"; tail -5 $M/train.log; touch $OUT/$TAG.DONE; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png > $M/render.log 2>&1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python $T/census.py $M/ckpt.pt
  CUDA_VISIBLE_DEVICES=$G python $T/score_split.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag $TAG 2>&1 | grep -avE "Warning|warn|torchvision|Loading model|Setting up"
  echo "$COMMON $*" > $M/train_args.txt
  rm -f $M/ckpt.pt
  echo "<<< $TAG DONE $(date +%H:%M)"; touch $OUT/$TAG.DONE; }

arm long45k_s101 --iters 45000 --refine_stop 38000 --noise_stop 38000 --lpips_from 30000 --seed 101
arm long50k      --iters 50000 --refine_stop 42000 --noise_stop 42000 --lpips_from 33000 --seed 42
echo "=== Q3A DONE $(date +%H:%M) ==="; touch $OUT/Q3A.DONE

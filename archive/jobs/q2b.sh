#!/bin/bash
# GPU1 follow-up. Waits for the tower sr01 arm to release the card (~23:00).
#
# ARM 1 -- churn_r29n29. The churn axis has three points now (15k/8k -> 25k/15k -> 25k/25k giving
# 0 -> +0.136 -> +0.302) and it is still climbing at the last one. 29k/29k is the boundary at
# iters=30000, so this settles whether 25k/25k is a peak or just the far end of what was sampled.
# ~1h35.
#
# ARM 2 -- long45k. If long60k (running on GPU0) wins, the optimum is between 30k and 60k and this
# locates it; if long60k loses, this says whether the loss is gradual (a real length limit) or a
# cliff (something else breaks past 45k). Either outcome is informative. ~2h20.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai; OUT=/mnt/d/avv/r43_bonsai; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
COMMON="--cap_max 5000000 --scale_reg 0.1"
G=1
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

arm churn_r29n29 --iters 30000 --refine_stop 29000 --noise_stop 29000 --lpips_from 12000 --seed 42
arm long45k      --iters 45000 --refine_stop 38000 --noise_stop 38000 --lpips_from 30000 --seed 42
echo "=== Q2B DONE $(date +%H:%M) ==="; touch $OUT/Q2B.DONE

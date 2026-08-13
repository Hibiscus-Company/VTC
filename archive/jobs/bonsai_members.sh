#!/bin/bash
# THE RELIABLE LEVER: bonsai 3 -> 6 members.
# bonsai is the shallowest ensemble we ship (3, vs chair 7 and towers 6-7), so it sits on the STEEP
# part of the 1/N curve. Composition is the only lever that has never transferred below 1x in 10
# graded rounds -- it is a variance-reduction argument, not a "clean up the render" argument, so it
# does NOT depend on the proxy/production noise gap that just burned us on the encode (r23).
# Reference marginals from our own graded rounds: tower 2->3 seeds +0.232, 3->4 +0.142 per scene.
# bonsai 3->6 therefore projects roughly +0.3..+0.4 ON THAT SCENE = +0.04..+0.06 blended.
#
# Recipe is r16's EXACT bonsai production config (aa no-UT antialiased + the capD churn-stop that
# fixed the bonsai collapse -- refine_stop 15k / noise_stop 8k). Existing members are seeds 42/7/13.
# Claim-based over both GPUs; each waits for its GPU to actually be free (no stacking).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2/bonsai
OUT=/mnt/d/avv/r24_bonsai
CLAIMS=$OUT/claims
mkdir -p "$CLAIMS"
SEEDS="101 202 303"
RECIPE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"

run_one() {
  local G=$1 S=$2 M=$OUT/aa$2
  conda activate gsplat
  echo ">>> gpu$G bonsai seed $S train $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $DATA/train --images images \
    --seed $S $RECIPE --out $M 2>&1 | tail -2 || { echo "!!! bonsai s$S TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $DATA/test/test_poses.csv --out $M/test_render --png_dir $M/test_png 2>&1 | tail -1 \
    || { echo "!!! bonsai s$S RENDER FAIL"; return 1; }
  rm -f $M/ckpt.pt
  echo "<<< gpu$G bonsai s$S done $(date +%H:%M) ($(ls $M/test_png | wc -l) pngs)"
  touch $OUT/aa${S}.DONE
}

gpu_free() {
  local m; m=$(nvidia-smi --id=$1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-99999}" -lt 2000 ]
}

worker() {
  local G=$1
  for S in $SEEDS; do
    if mkdir "$CLAIMS/$S" 2>/dev/null; then
      while ! gpu_free "$G"; do sleep 60; done
      run_one "$G" "$S"
    fi
  done
}

worker 0 &
worker 1 &
wait
echo "=== BONSAI MEMBERS DONE $(date) ==="
touch /mnt/d/avv/bonsai_members.DONE

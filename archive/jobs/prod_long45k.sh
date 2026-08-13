#!/bin/bash
# PRODUCTION bonsai members on the long45k recipe, full 248-image train set, test poses.
# GPU passed as $1. Seeds passed as $2..
#
# WHY NOW: the pre-registered encoded k=2 paired gate PASSED on long45k --
#     ref 71.7394 -> long45k 72.6437,  paired +0.9043, sd 2.0617, t = +2.321, 16/28 wins
#     forecast scene +0.7687 -> **LB TOTAL +0.1098**, i.e. 8.9x r36 and 5.5x r35.
# Two seeds agree on the raw split too (+0.7117 / +0.8063, mean +0.759, spread 0.095), and the
# length curve is unimodal with 45k at its peak (30k +0.322, 45k +0.759, 50k +0.523, 60k +0.257).
# These members are what r37 will be built from.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
PRI=/mnt/d/avv/data/phase1/private_set2; OUT=/mnt/d/avv/r45_prod; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
mkdir -p $OUT
BASE="--iters 45000 --cap_max 5000000 --refine_stop 38000 --noise_stop 38000 --lpips_from 30000 --scale_reg 0.1"
G=$1; shift
while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits|tr -d ' ')" -gt 3000 ]; do sleep 60; done
for SD in "$@"; do
  M=$OUT/s$SD
  [ -f $OUT/s$SD.DONE ] && { echo "s$SD done, skip"; continue; }
  mkdir -p $M; conda activate gsplat
  echo ">>> prod long45k seed $SD (gpu$G) START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $PRI/bonsai/train --images images \
    --out $M $BASE --seed $SD > $M/train.log 2>&1
  grep -aE "^Saved" $M/train.log | tail -1
  [ -f $M/ckpt.pt ] || { echo "!!! seed $SD NO CKPT"; tail -5 $M/train.log; continue; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $PRI/bonsai/test/test_poses.csv --out $M/test_render --png_dir $M/test_png > $M/render.log 2>&1
  conda activate fastgs2; CUDA_VISIBLE_DEVICES=$G python $T/census.py $M/ckpt.pt
  echo "$BASE --seed $SD" > $M/train_args.txt
  rm -f $M/ckpt.pt
  n=$(ls $M/test_png 2>/dev/null | wc -l)
  [ "$n" -eq 28 ] || { echo "!!! seed $SD wrote $n pngs, expected 28"; continue; }
  echo "<<< prod seed $SD DONE $(date +%H:%M) ($n pngs)"; touch $OUT/s$SD.DONE
done
echo "=== PROD gpu$G DONE $(date +%H:%M) ==="; touch $OUT/G$G.DONE

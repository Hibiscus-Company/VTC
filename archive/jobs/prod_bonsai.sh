#!/bin/bash
# PRODUCTION bonsai members at the gate-passed scale_reg 0.1, FULL 248 frames, test poses.
# Gate evidence: diversity-matched k=2 A/A pairs, ship encode, +0.1653 t=+2.12 17/28.
# Single-model replicates: lam0.1 {71.9911, 71.9816, 71.8254} mean 71.933
#                          lam0.01 {71.9030, 71.6951}        mean 71.799   -> +0.134
# Runs seeds back to back on whichever GPU frees; each ckpt deleted after render (disk 17 GB).
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
PRI=/mnt/d/avv/data/phase1/private_set2; OUT=/mnt/d/avv/r38_prod; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
mkdir -p $OUT
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 --scale_reg 0.1"
one(){ local G=$1 SD=$2 M=$OUT/s$2
  [ -f $OUT/s$SD.DONE ] && return 0
  mkdir -p $M; conda activate gsplat
  echo ">>> prod seed $SD (gpu$G) START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $PRI/bonsai/train --images images \
    --out $M $BASE --seed $SD 2>&1 | grep -aE "Saved|Error|error" | tail -2
  [ -f $M/ckpt.pt ] || { echo "!!! seed $SD NO CKPT"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $PRI/bonsai/test/test_poses.csv --out $M/test_render --png_dir $M/test_png 2>&1|tail -1
  conda activate fastgs2; python $T/census.py $M/ckpt.pt
  echo "$BASE --seed $SD" > $M/train_args.txt
  rm -f $M/ckpt.pt
  echo "<<< prod seed $SD DONE $(date +%H:%M) ($(ls $M/test_png 2>/dev/null|wc -l) pngs)"
  touch $OUT/s$SD.DONE; }
worker(){ local G=$1; shift; for SD in "$@"; do
    while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]; do sleep 60; done
    one $G $SD; done; }
worker 1 555 777 999 > $OUT/gpu1.log 2>&1 &
worker 0 111 222 333 > $OUT/gpu0.log 2>&1 &
wait; echo "=== PROD DONE $(date +%H:%M) ==="; touch $OUT/ALL.DONE

#!/bin/bash
# Remaining arms. Census now a separate file (the in-function heredoc silently failed and we
# lost aniso001's shape numbers because the ckpt was deleted right after). Keep ckpt until AFTER
# the census prints.
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai; OUT=/mnt/d/avv/r36_shape; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 --seed 42"
arm(){
  local G=$1 TAG=$2 M=$OUT/$2; shift 2; local EX="$*"
  mkdir -p $M; conda activate gsplat
  echo ">>> $TAG (gpu$G) START $(date +%H:%M) [$EX]"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --out $M $BASE $EX 2>&1 | grep -aE "Baked|Saved|Error|error" | tail -3
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT"; touch $OUT/$TAG.DONE; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  python $T/census.py $M/ckpt.pt
  python $T/bar.py $M/eval_png
  rm -f $M/ckpt.pt
  echo "<<< $TAG DONE $(date +%H:%M)"; touch $OUT/$TAG.DONE
}
arm 1 sr01 --scale_reg 0.1 > $OUT/sr01.log 2>&1 &
while [ ! -f $OUT/sr001seed42.DONE ]; do sleep 45; done
arm 0 sr03 --scale_reg 0.3 > $OUT/sr03.log 2>&1 &
wait
arm 0 regstop15k --reg_stop 15000 > $OUT/regstop15k.log 2>&1
echo "=== QUEUE3 DONE $(date +%H:%M) ==="; touch $OUT/ALL.DONE

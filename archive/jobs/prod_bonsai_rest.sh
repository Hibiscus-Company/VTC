#!/bin/bash
# Finish the 3 bonsai production members that prod_bonsai.sh never got to (222, 333, 999) --
# they were dropped when the prod queue was killed to hand GPU1 to UBS.
#
# WHY now, and why this is not a new bet: r35 GRADED 77.7106 (+0.0199 over r32, all three
# metrics up). r35's only change was bonsai = 7 old members + 3 new scale_reg=0.1 members.
# So the LB has now CONFIRMED the scale_reg=0.1 direction. Going 3 new -> 6 new shifts the
# ensemble mean further onto the members the LB just paid for, and it also unlocks a real
# composition sweep (drop the weakest OLD members, which the 7/3 split could not test).
#
# GPU0 ONLY. UBS owns GPU1 -- do not race it (that race already cost us ~10 min once).
# Identical recipe to s111/s555/s777, seed being the only difference.
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
PRI=/mnt/d/avv/data/phase1/private_set2; OUT=/mnt/d/avv/r38_prod; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
mkdir -p $OUT
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 --scale_reg 0.1"
one(){ local G=$1 SD=$2 M=$OUT/s$2
  [ -f $OUT/s$SD.DONE ] && { echo "s$SD already done, skip"; return 0; }
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
  local n=$(ls $M/test_png 2>/dev/null|wc -l)
  [ "$n" -eq 28 ] || { echo "!!! seed $SD wrote $n pngs, expected 28"; return 1; }
  echo "<<< prod seed $SD DONE $(date +%H:%M) ($n pngs)"
  touch $OUT/s$SD.DONE; }
for SD in 222 333 999; do
  while [ "$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]; do sleep 60; done
  one 0 $SD
done
echo "=== PROD REST DONE $(date +%H:%M) ==="; touch $OUT/REST.DONE

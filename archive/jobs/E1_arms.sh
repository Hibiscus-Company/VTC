#!/bin/bash
# GPU0 = E1: per-image affine appearance IN THE LOSS (--app_affine). Never used in any shipped
#   run. The prior "per-image exposure is dead" was measured on TOWERS (outdoor drone, matte
#   roofing, stable exposure) and as a POST-HOC correction, not as train-time damage -- the same
#   structure as k1, where the post-hoc warp recovered only 26% of the oracle.
#   KILL CRITERION built into the code: it prints |M-I| mean/max and saves app_affine.pt. If the
#   learned transforms come out ~identity, the drift is not hurting and the branch dies with no
#   scoring needed.
# GPU1 = bank the gate-passed scale_reg gain: first production member at lam=0.1, seed 101.
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
ES=/mnt/d/avv/evalsplit/bonsai; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"
run(){ local G=$1 TAG=$2 M=/mnt/d/avv/r38/$2; shift 2
  mkdir -p $M; conda activate gsplat
  echo ">>> $TAG (gpu$G) START $(date +%H:%M) [$*]"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --out $M $BASE "$@" 2>&1 | grep -aE "Saved|app_affine|Error|error" | tail -4
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT"; touch /mnt/d/avv/r38/$TAG.DONE; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/render --png_dir $M/eval_png 2>&1|tail -1
  conda activate fastgs2; python $T/census.py $M/ckpt.pt; python $T/bar.py $M/eval_png
  rm -f $M/ckpt.pt; echo "<<< $TAG DONE $(date +%H:%M)"; touch /mnt/d/avv/r38/$TAG.DONE; }
mkdir -p /mnt/d/avv/r38
run 0 appaffine --seed 42 --app_affine > /mnt/d/avv/r38/appaffine.log 2>&1 &
run 1 sr01_s101 --seed 101 --scale_reg 0.1 > /mnt/d/avv/r38/sr01_s101.log 2>&1 &
wait; echo "=== DONE $(date +%H:%M) ==="; touch /mnt/d/avv/r38/ALL.DONE

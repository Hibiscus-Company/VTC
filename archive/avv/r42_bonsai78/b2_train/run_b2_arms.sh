#!/bin/bash
# ============================================================================================
# B2 arm list: the TRAINING-side attack on bonsai's in-sample blur floor.
#
# DEPLOY FIRST (never touch the live trainer -- a job is running against this repo):
#     cp /mnt/d/avv/r42_bonsai78/b2_train/train_gsplat_b2.py  gsplat_track/
#     cp /mnt/d/avv/r42_bonsai78/b2_train/render_gsplat_b2.py gsplat_track/
# Both are NEW filenames; gsplat_track/train_gsplat.py and render_gsplat.py are unmodified.
# With every new flag left at its default the patched trainer is numerically identical to the
# original (the diff replaces only 4 lines, each by `1.0 * <same expression>`).
#
# WAVES. Wave G is a KILL-TEST and gates everything after it. Do not launch W or B until G
# has read out; if the gate is null they are all dead and you save ~10 GPU-h.
#
# All arms: recipe base verbatim, single variable, eval split, scored by bar.py (BAR 71.799,
# best single 71.9911 = r36_shape/sr01). 1-v-1 training-noise floor on this split = 0.407.
# ============================================================================================
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai
B2=/mnt/d/avv/r42_bonsai78/b2_train
OUT=/mnt/d/avv/r42_bonsai78/b2_runs
mkdir -p $OUT
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 \
--lpips_from 12000 --scale_reg 0.1 --seed 42"
SIDE="--sharp_csv $B2/bonsai_sharp_sidecar.csv --sharp_stat log_lapvar"

arm(){ # $1 gpu  $2 tag  $3.. extra args
  local G=$1 T=$2 M=$OUT/$2; shift 2; local EX="$*"
  mkdir -p $M
  conda activate gsplat
  echo ">>> $T (gpu$G) START $(date +%H:%M)  [$EX]"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat_b2.py \
    --source $ES/train_sub --images images --out $M $BASE $EX 2>&1 \
    | grep -aE "view_subset|sharpness weights|blur sigma init|blur_view|WARNING|sigma\[|Saved|Error|error" \
    | tail -40
  [ -f $M/ckpt.pt ] || { echo "!!! $T NO CKPT"; return 1; }
  # render SHARP (blur_mode none). The test-time sigma policy is a per-image operator on the
  # saved PNGs -- sweep it offline with b3_sigma_sweep.py, never by re-rendering.
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat_b2.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  python /home/bkai/.claude/jobs/1c9cf7e9/tmp/bar.py $M/eval_png
  # ALWAYS split first8 / last20: A3 measured a 13.98-point sub-population gap and a
  # scene-averaged number hides it.
  python $B2/score_split.py $M/eval_png
  rm -f $M/ckpt.pt          # keep blur.pt
  echo "<<< $T DONE $(date +%H:%M)"; touch $OUT/$T.DONE
}

case "${1:-help}" in

# ---------------------------------------------------------------- WAVE G: the kill-test
# Coverage-MATCHED sharp-vs-blurry training subsets. Every block of 6 consecutive frames
# contributes exactly 3 views to EVERY arm, so view count (108), mean inter-view gap (25.0
# frames) and camera-center coverage are identical; the ONLY difference is a 0.62-nat
# (= 0.62 px equivalent sigma, 0.85 sd) shift in photo sharpness.
# READ: dScore(G_sharp - G_blurry). Neither arm is comparable to the 220-view bar.
# PASS if >= +0.8 (2 noise floors). NULL if |d| < 0.4 -> (1) and (2) are both dead, stop.
g)  arm 0 G_sharp  --view_subset $B2/subsets/b6m3_sharp.txt  > $OUT/G_sharp.log  2>&1 &
    arm 1 G_blurry --view_subset $B2/subsets/b6m3_blurry.txt > $OUT/G_blurry.log 2>&1 &
    wait
    arm 0 G_rand   --view_subset $B2/subsets/b6m3_rand.txt   > $OUT/G_rand.log   2>&1   # midpoint / monotonicity
    ;;

# ---------------------------------------------------------------- WAVE W: (1) loss weighting
# Dose-response on the exponential tilt, plus the coverage-preserving hard mask.
w)  arm 0 W_tilt05 $SIDE --sw_mode tilt --sw_beta 0.5 --sw_scope photo > $OUT/W_tilt05.log 2>&1 &
    arm 1 W_tilt10 $SIDE --sw_mode tilt --sw_beta 1.0 --sw_scope photo > $OUT/W_tilt10.log 2>&1 &
    wait
    arm 0 W_topk50 $SIDE --sw_mode topk --sw_topk 0.5 --sw_topk_stratify 6 \
                         --sw_scope photo > $OUT/W_topk50.log 2>&1 &
    arm 1 W_tilt05all $SIDE --sw_mode tilt --sw_beta 0.5 --sw_scope all > $OUT/W_tilt05all.log 2>&1 &
    wait ;;

# ---------------------------------------------------------------- WAVE B: (2) per-view blur
# B_smoke FIRST (6k iters, ~18 min): reads the sigma trajectory before committing 5.6 GPU-h.
bsmoke) conda activate gsplat
    CUDA_VISIBLE_DEVICES=${2:-0} python gsplat_track/train_gsplat_b2.py \
      --source $ES/train_sub --images images --out $OUT/B_smoke \
      --iters 6000 --cap_max 5000000 --refine_stop 4000 --noise_stop 2000 \
      --lpips_from 99999 --scale_reg 0.1 --seed 42 $SIDE \
      --blur_view learn --blur_ref_q 0.75 --blur_lr 1e-2 --blur_warmup 1000 \
      --lambda_blur 0.02 2>&1 | grep -aE "blur|sigma\[|Saved" | tail -20 ;;

b)  arm 0 B_frozen $SIDE --blur_view frozen --blur_ref_q 0.75 > $OUT/B_frozen.log 2>&1 &
    arm 1 B_learn  $SIDE --blur_view learn --blur_ref_q 0.75 --blur_lr 1e-2 \
                         --lambda_blur 0.02 --blur_reg init > $OUT/B_learn.log 2>&1 &
    wait
    arm 0 B_learn_sm $SIDE --blur_view learn --blur_ref_q 0.75 --blur_lr 1e-2 \
                         --lambda_blur 0.02 --lambda_blur_smooth 0.1 > $OUT/B_learn_sm.log 2>&1 &
    arm 1 B_late   $SIDE --blur_view learn --blur_ref_q 0.75 --blur_lr 1e-2 \
                         --lambda_blur 0.02 --blur_from 15000 > $OUT/B_late.log 2>&1 &
    wait ;;

# ---------------------------------------------------------------- (3) sigma_test: ZERO GPU
# Blurring is a per-image operator on the saved PNGs, so the whole policy is swept offline.
s)  conda activate fastgs2
    for M in $OUT/B_frozen $OUT/B_learn $OUT/B_learn_sm $OUT/B_late; do
      [ -d $M/eval_png ] || continue
      echo "=== $M"
      CUDA_VISIBLE_DEVICES="" PYTHONPATH=/mnt/c/Users/BKAI/an_plaza2/FastGS \
        python $B2/b3_sigma_sweep.py --render $M/eval_png --blur_pt $M/blur.pt
    done ;;

*)  echo "usage: $0 {g|w|bsmoke|b|s}"; echo
    echo "  g       WAVE G kill-test   2 parallel + 1  ~4.2 GPU-h wall ~2.8 h"
    echo "  w       WAVE W weighting   4 arms          ~5.6 GPU-h wall ~2.8 h"
    echo "  bsmoke  6k smoke on blur   1 arm           ~0.3 GPU-h"
    echo "  b       WAVE B blur        4 arms          ~5.8 GPU-h wall ~2.9 h"
    echo "  s       (3) sigma sweep    CPU only        0 GPU-h, ~6 min/arm"
    ;;
esac

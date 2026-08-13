#!/bin/bash
# B3 top-5 untried-axis arms for bonsai, eval split, all SINGLE-VARIABLE vs the shipped recipe.
#
# BARS (all on today's scorer, all PNG, all 28 eval holes at /mnt/d/avv/evalsplit/bonsai/eval_gt):
#   K1 matched-seed control (this exact BASE, seed 42) ......... 71.9029
#   scale_reg 0.1 3-replicate mean (71.9911/71.9816/71.8254) ... 71.9327   sd 0.0930  SE 0.0537
#   best single arm ever measured (tw1.0, same BASE + tw) ...... 72.0606
#   6-arm PNG ensemble mean .................................... 72.2644
# Gate a new arm against 71.9327 +- 0.107 (1 sigma for single-arm-vs-3-replicate-mean),
# NOT against the 0.407 1-v-1 floor (that floor predates the 3 replicates).
#
# NOTE both GPUs were at 99-100% / 15.7 of 16.3 GiB when this was written
# (r39_ubs/cap5M on gpu0, r41_towersr/sr001 on gpu1). Nothing here can start until one frees.
# NOTE /mnt/d had 16 GB free. Each 5M ckpt is 1.18 GB -> the rm below is mandatory.
# Reclaimable now: /mnt/d/avv/r33_bonsai/eval_s42/ckpt.pt is 1.888 GB of dead 8M arm.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai
OUT=/mnt/d/avv/r42_bonsai78/b3_arms
mkdir -p $OUT
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 --seed 42"

arm(){ # $1 gpu  $2 tag  $3.. extra args
  local G=$1 T=$2 M=$OUT/$2; shift 2; local EX="$*"
  mkdir -p $M
  conda activate gsplat
  echo ">>> $T (gpu$G) START $(date +%H:%M)  [$EX]"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --out $M $EX 2>&1 \
    | grep -aE "Baked|Saved|Error|error" | tail -3
  [ -f $M/ckpt.pt ] || { echo "!!! $T NO CKPT"; return 1; }
  conda activate fastgs2
  python /home/bkai/.claude/jobs/1c9cf7e9/tmp/b3/census.py $M/ckpt.pt   # collapse gate + live-count
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  python /home/bkai/.claude/jobs/1c9cf7e9/tmp/bar.py $M/eval_png
  echo "$EX" > $M/train_args.txt
  rm -f $M/ckpt.pt            # MANDATORY, 1.18 GB each and /mnt/d has 16 GB
  echo "<<< $T DONE $(date +%H:%M)"; touch $OUT/$T.DONE
}

# ---------------------------------------------------------------- #1  ~0.5 GPU-h
# Warm-start LPIPS-only second stage (exp26 class; BEST TRACK B arm on HCM0181, never run on
# bonsai). Reuses the surviving 71.9029 ckpt, so it costs 8k iters not 30k. No relocation, no
# reg, LPIPS from step 1: attacks the IN-SAMPLE fit, which is 84% of bonsai's held-out error.
# GATE THIS ONE AGAINST 71.9029 (its own base), not 71.9327.
arm 0 warmpolish --init_ckpt /mnt/d/avv/bonsai_eval/K1_noUT_aa/ckpt.pt \
  --iters 8000 --cap_max 5000000 --refine_stop 0 --noise_stop 0 \
  --opacity_reg 0 --scale_reg 0 --lambda_lpips 0.2 --lpips_from 0 --means_lr 4.8e-5 --seed 42

# ---------------------------------------------------------------- #2  ~1.3 GPU-h
# The only two single-variable POSITIVE knobs on this recipe, never combined, and their metric
# signatures are complementary: scale_reg 0.1 = cP -0.021 cS -0.075 cL +0.180 (buys LPIPS);
# texture_weight 1.0 = cP +0.240 cS +0.063 cL -0.148 (buys PSNR/SSIM). Additive prediction 72.145.
arm 0 tw1_sr01 $BASE --scale_reg 0.1 --texture_weight 1.0

# ---------------------------------------------------------------- #3  ~2.9 GPU-h (2 arms)
# lambda_lpips at FIXED lpips_from. bonsai's entire deficit is LPIPS (0.2448 vs the towers'
# 0.109-0.132) and the metric weights it 40%, while the loss weights it 0.1 against L1's 0.8.
# The only prior bonsai test of a bigger lambda also moved lpips_from AND ran on the retired UT
# recipe AND on the pre-drift scorer, so this knob has never actually been measured here.
arm 0 lp025 $BASE --scale_reg 0.1 --lambda_lpips 0.25
arm 1 lp050 $BASE --scale_reg 0.1 --lambda_lpips 0.5

# ---------------------------------------------------------------- #4  ~2.6 GPU-h (2 arms)
# opacity_reg, NEVER run on bonsai. Measured today: at 5M, 63.2% of bonsai's gaussians end below
# opacity 0.005 and only 21.1% above 0.05 -- roughly 3x the tower's dead fraction and far outside
# the 27-47% the flag's own help text cites. Diagnosed mechanism (audit r6) is the constant
# opacity_reg pull, which keeps running for the 15k steps AFTER refine_stop switches off the
# relocation that would recycle them. exp21's counter-evidence (HCM0181, -0.22) failed by
# "more train capacity, worse generalisation" -- the one failure bonsai can afford.
arm 0 opr0002 $BASE --scale_reg 0.1 --opacity_reg 0.002
arm 1 opr0    $BASE --scale_reg 0.1 --opacity_reg 0.0

# ---------------------------------------------------------------- #5  ~1.35 GPU-h
# Churn. refine_stop 15000 / noise_stop 8000 were chosen mid-incident on 17/07 under the retired
# UT recipe and never revisited across two recipe generations. Relocation is the ONLY mechanism
# that recycles the dead reservoir and it is switched off at the halfway point.
# HALF-STEP ONLY (22k, not 25k): this is the knob whose mis-setting produced the fog collapse
# (r10->r10b was worth +25.3 on the scene). The census line above is the collapse gate --
# abort if median opacity < 0.02 or dead_frac > 0.75.
arm 0 rs22k --iters 30000 --cap_max 5000000 --refine_stop 22000 --noise_stop 8000 \
  --lpips_from 12000 --seed 42 --scale_reg 0.1

echo "=== B3 ARMS DONE $(date +%H:%M) ==="; touch $OUT/ALL.DONE

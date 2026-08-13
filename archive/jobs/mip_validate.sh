#!/bin/bash
# Track A: validate the Mip 3D filter on EVAL-SPLIT (test poses have no GT, so decide here).
# For chair + HCM0181 tower: filter the eval ckpt -> render eval holes -> score (a) standalone
# vs the un-filtered baseline, and (b) as an ADDED member to the existing eval ensemble. If the
# ensemble-add helps on eval, Track A graduates to filtering all r20 test members. Gated on
# r20 build (=> GPU1 free) to avoid stacking on a production trainer.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
G=1

while [ ! -f /mnt/d/avv/r20_combined.BUILT ]; do sleep 60; done
echo "=== r20 built, GPU1 free -- Mip eval validation $(date) ==="

# chair: eval ckpt = tw_test/chair_ema099 (baseline eval 69.8051), native render
# tower: eval ckpt = tw_test/HCM0181_ema999 (baseline eval 75.4293), UT native render
run() {  # $1 scene $2 ckptdir $3 sparse $4 renderflag $5 evalgt
  local SC=$1 M=$2 SP=$3 RF=$4 GT=$5 O=/mnt/d/avv/mip/$1
  mkdir -p $O
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python scripts/mip3d_filter.py \
    --ckpt $M/ckpt.pt --sparse $SP --out $O/mip_ckpt.pt --filter_scale 0.2 2>&1 | tail -1 \
    || { echo "!!! [$SC] FILTER FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $O/mip_ckpt.pt --csv /mnt/d/avv/evalsplit/$SC/eval_poses.csv \
    --out $O/eval_render --png_dir $O/eval_png $RF 2>&1 | tail -1 || { echo "!!! [$SC] RENDER FAIL"; return 1; }
  conda activate fastgs2
  echo "--- [$SC] Mip standalone (vs its own baseline) ---"
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $O/eval_png --gt_dir $GT --tag "${SC}_mip_solo"
}

run chair /mnt/d/avv/tw_test/chair_ema099 /mnt/d/avv/evalsplit/chair/train_sub/sparse/0 "" /mnt/d/avv/evalsplit/chair/eval_gt
run HCM0181 /mnt/d/avv/tw_test/HCM0181_ema999 /mnt/d/avv/evalsplit/HCM0181/train_sub/sparse/0 "--ut_render native" /mnt/d/avv/evalsplit/HCM0181/eval_gt

# ensemble-add test: does adding the Mip member to the existing eval ensemble help?
conda activate fastgs2
echo "=== ENSEMBLE-ADD (Mip member joins existing eval ensemble) ==="
CUDA_VISIBLE_DEVICES=$G python scripts/combiner_sweep.py --tag chair_mipadd --out_root /mnt/d/avv/mip \
  --dirs /mnt/d/avv/tw_test/chair_ema099/eval_png /mnt/d/avv/tw_test/chair_ema999/eval_png \
         /mnt/d/avv/tw_test/chair_capmax2M/eval_png /mnt/d/avv/tw_test/chair_aniso01/eval_png \
         /mnt/d/avv/mip/chair/eval_png \
  --gt_dir /mnt/d/avv/evalsplit/chair/eval_gt 2>/dev/null | grep " mean "
CUDA_VISIBLE_DEVICES=$G python scripts/combiner_sweep.py --tag tower_mipadd --out_root /mnt/d/avv/mip \
  --dirs /mnt/d/avv/tw_test/HCM0181_ema999/eval_png /mnt/d/avv/tw_test/HCM0181_ema099/eval_png \
         /mnt/d/avv/tw_test/HCM0181_minop02/eval_png /mnt/d/avv/tw_test/HCM0181_skydome50k/eval_png \
         /mnt/d/avv/mip/HCM0181/eval_png \
  --gt_dir /mnt/d/avv/evalsplit/HCM0181/eval_gt 2>/dev/null | grep " mean "
echo "REFERENCE (no-Mip means): chair 71.0682 | tower 76.3883"
echo "=== MIP VALIDATION DONE $(date) ==="
touch /mnt/d/avv/mip_validate.DONE

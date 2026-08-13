#!/bin/bash
# Retrain the perceptual-pass winner pC (30k, capD churn-fix, lpips_from 12k) on FULL bonsai
# (248 frames), 2 seeds SEQUENTIALLY on one GPU (other GPU busy with chair eval), ensemble,
# no field. Output feeds the round12 bundle.
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
G=$1
S=/mnt/d/avv/data/phase1/private_set2/bonsai/train
CSV=/mnt/d/avv/data/phase1/private_set2/bonsai/test/test_poses.csv
OUT=/mnt/d/avv/bonsai_pc
ARGS="--cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --iters 30000 --lpips_from 12000"
mkdir -p $OUT
conda activate gsplat
for SD in 42 7; do
  M=$OUT/ut$SD
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $S --images images --ut --seed $SD --out $M $ARGS 2>&1 | tail -2
  echo "$ARGS" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $CSV --out $M/test_render --png_dir $M/test_png \
    --ut_render native 2>&1 | tail -1
done
conda activate fastgs2
python ensemble_renders.py --dirs $OUT/ut42/test_png $OUT/ut7/test_png \
  --weights 0.5 0.5 --masks none none \
  --out $OUT/ens/jpg --png_dir $OUT/ens/png --names_from $CSV
echo "=== BONSAI pC FINAL READY: $OUT/ens/png ($(ls $OUT/ens/png | wc -l) imgs) ==="

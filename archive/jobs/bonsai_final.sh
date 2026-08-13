#!/bin/bash
# Retrain the EVAL-WINNING bonsai recipe on the FULL 248-frame train set (2 seeds),
# render test poses, ensemble (no field -- bonsai field failed held-out x-val), then
# splice into leg-1's towers/chair to build round10b.
# Usage: bonsai_final.sh "<train-args>"   e.g. "--cap_max 1000000 --refine_stop 15000 --noise_stop 15000"
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
ARGS="$1"
S=/mnt/d/avv/data/phase1/private_set2/bonsai/train
CSV=/mnt/d/avv/data/phase1/private_set2/bonsai/test/test_poses.csv
OUT=/mnt/d/avv/bonsai_final
mkdir -p $OUT

seed_run() {  # $1 gpu  $2 seed
  local G=$1 SD=$2
  local M=$OUT/ut$SD
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $S --images images --ut --seed $SD \
    --out $M --iters 30000 --lpips_from 25000 $ARGS 2>&1 | tail -2
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $CSV --out $M/test_render --png_dir $M/test_png \
    --ut_render native 2>&1 | tail -1
}
( seed_run 0 42 ) &
( seed_run 1 7 ) &
wait

conda activate fastgs2
# 2-seed mean ensemble, NO field (bonsai field zeroed: held-out x-val +0.003 = junk)
python ensemble_renders.py --dirs $OUT/ut42/test_png $OUT/ut7/test_png \
  --weights 0.5 0.5 --masks none none \
  --out $OUT/ens/jpg --png_dir $OUT/ens/png \
  --names_from $CSV
echo "=== BONSAI FINAL renders ready: $OUT/ens/png ==="
ls $OUT/ens/png | wc -l

# round10b: leg-1 towers/chair + fixed bonsai
python build_submission_zip.py \
  --scene_dirs HCM0421=/mnt/d/avv/r2r8/ens/HCM0421/png \
               HCM0539=/mnt/d/avv/r2r8/ens/HCM0539/png \
               HCM0540=/mnt/d/avv/r2r8/ens/HCM0540/png \
               HCM0644=/mnt/d/avv/r2r8/ens/HCM0644/png \
               HCM0674=/mnt/d/avv/r2r8/ens/HCM0674/png \
               chair=/mnt/d/avv/r2r8/ens/chair/png \
               bonsai=$OUT/ens/png \
  --data_root /mnt/d/avv/data/phase1/private_set2 \
  --out /mnt/d/avv/submissions/sub_round10b_bonsaifix.zip
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round10b_bonsaifix.zip \
  --data_root /mnt/d/avv/data/phase1/private_set2
echo "=== ROUND10b BUILT: sub_round10b_bonsaifix.zip ==="

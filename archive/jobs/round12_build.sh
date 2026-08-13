#!/bin/bash
# ROUND12 unattended chain:
#   1. chair lpearly (eval winner 68.54 vs base 67.79) retrained on FULL 205 frames,
#      2 seeds in parallel (one per GPU), render test poses, ensemble NO field
#   2. compose round12 = r11 towers (60k/8M+field) + chair-lpearly + bonsai-pC -> zip + verify
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
S=/mnt/d/avv/data/phase1/private_set2/chair/train
CSV=/mnt/d/avv/data/phase1/private_set2/chair/test/test_poses.csv
OUT=/mnt/d/avv/chair_lpearly
ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
mkdir -p $OUT

seed_run() {
  local G=$1 SD=$2
  local M=$OUT/ut$SD
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $S --images images --ut --seed $SD --out $M $ARGS 2>&1 | tail -2
  echo "$ARGS" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $CSV --out $M/test_render --png_dir $M/test_png \
    --ut_render native 2>&1 | tail -1
}
( seed_run 0 42 ) &
( seed_run 1 7 ) &
wait

conda activate fastgs2
python ensemble_renders.py --dirs $OUT/ut42/test_png $OUT/ut7/test_png \
  --weights 0.5 0.5 --masks none none \
  --out $OUT/ens/jpg --png_dir $OUT/ens/png --names_from $CSV || { echo "ROUND12 ENSEMBLE FAILED"; exit 1; }
echo "=== CHAIR lpearly FINAL READY ($(ls $OUT/ens/png | wc -l) imgs) ==="

PRI=/mnt/d/avv/data/phase1/private_set2
python build_submission_zip.py --scene_dirs \
  HCM0421=/mnt/d/avv/r2r9/ens/HCM0421/png \
  HCM0539=/mnt/d/avv/r2r9/ens/HCM0539/png \
  HCM0540=/mnt/d/avv/r2r9/ens/HCM0540/png \
  HCM0644=/mnt/d/avv/r2r9/ens/HCM0644/png \
  HCM0674=/mnt/d/avv/r2r9/ens/HCM0674/png \
  chair=$OUT/ens/png \
  bonsai=/mnt/d/avv/bonsai_pc/ens/png \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round12_videofix.zip
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round12_videofix.zip --data_root $PRI
cat > /mnt/d/avv/submissions/sub_round12_videofix.PROVENANCE.txt <<'EOF'
sub_round12_videofix.zip — per-scene recipes (NOT uniform):
  HCM0421/0539/0540/0644/0674: UT 60k/8M 2-seed + DIS lens field (same as round11, r2r9)
  chair: UT 60k/8M lpips_from 30k ("lpearly", eval 68.54 vs base 67.79) 2-seed, NO field
  bonsai: UT 30k capD+lpearly ("pC": cap 5M, refine 15k, noise 8k, lpips_from 12k,
          eval 71.36 vs capD 71.16) 2-seed, NO field
  !! video scenes deliberately NOT the tower recipe (bonsai collapses on late churn).
EOF
echo "=== ROUND12 BUILT: sub_round12_videofix.zip ==="

#!/bin/bash
# ROUND14 unattended chain — M1 (antialiased/no-UT) winners to full-train + video-scene swap:
#   bonsai: pC schedule minus --ut (K1 eval 71.90 vs 71.36), 30k, 2 seeds
#   chair:  lpearly schedule minus --ut (K3 eval 69.37 vs 68.54), 60k, 2 seeds
#   compose round14 = round13 towers + new chair + new bonsai -> zip + verify
# GPU1 branch first waits for kill_tests.sh (K4_seed1234) to finish.
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
BON_ARGS="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"
CHA_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"

t_r() {  # $1 gpu $2 scene $3 outdir $4 seed  rest=args   (NO --ut = antialiased mode)
  local G=$1 SC=$2 M=$3 SD=$4; shift 4
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $PRI/$SC/train --images images --seed $SD --out $M "$@" 2>&1 | tail -2
  echo "noUT-antialiased $*" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $PRI/$SC/test/test_poses.csv \
    --out $M/test_render --png_dir $M/test_png 2>&1 | tail -1
}

( t_r 0 chair  /mnt/d/avv/r14/chair_aa42  42 $CHA_ARGS
  t_r 0 bonsai /mnt/d/avv/r14/bonsai_aa42 42 $BON_ARGS
  t_r 0 bonsai /mnt/d/avv/r14/bonsai_aa7   7 $BON_ARGS ) &
( while pgrep -f "kill_tests.sh" >/dev/null; do sleep 60; done
  t_r 1 chair  /mnt/d/avv/r14/chair_aa7    7 $CHA_ARGS ) &
wait

conda activate fastgs2
python ensemble_renders.py --dirs /mnt/d/avv/r14/chair_aa42/test_png /mnt/d/avv/r14/chair_aa7/test_png \
  --weights 0.5 0.5 --masks none none \
  --out /mnt/d/avv/r14/chair_ens/jpg --png_dir /mnt/d/avv/r14/chair_ens/png \
  --names_from $PRI/chair/test/test_poses.csv || { echo "R14 CHAIR ENS FAILED"; exit 1; }
python ensemble_renders.py --dirs /mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
  --weights 0.5 0.5 --masks none none \
  --out /mnt/d/avv/r14/bonsai_ens/jpg --png_dir /mnt/d/avv/r14/bonsai_ens/png \
  --names_from $PRI/bonsai/test/test_poses.csv || { echo "R14 BONSAI ENS FAILED"; exit 1; }

python build_submission_zip.py --scene_dirs \
  HCM0421=/mnt/d/avv/r13/HCM0421/png HCM0539=/mnt/d/avv/r13/HCM0539/png \
  HCM0540=/mnt/d/avv/r13/HCM0540/png HCM0644=/mnt/d/avv/r13/HCM0644/png \
  HCM0674=/mnt/d/avv/r13/HCM0674/png \
  chair=/mnt/d/avv/r14/chair_ens/png \
  bonsai=/mnt/d/avv/r14/bonsai_ens/png \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round14_videoaa.zip
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round14_videoaa.zip --data_root $PRI
cat > /mnt/d/avv/submissions/sub_round14_videoaa.PROVENANCE.txt <<'EOF'
sub_round14_videoaa.zip — per-scene recipes:
  towers: SAME as round13 (family ensemble 2 UT @0.3 + 3 gates @0.1333 + field)
  chair:  ANTIALIASED no-UT 60k/8M lpearly (K3 eval 69.37 vs 68.54), 2-seed, no field
  bonsai: ANTIALIASED no-UT 30k capD+lpearly (K1 eval 71.90 vs 71.36), 2-seed, no field
  !! video scenes: NEVER add --ut back (k1=0, UT forces classic = loses antialiasing);
  !! bonsai: NEVER extend churn (collapse) or iters past 30k (reverses).
  vs r13 only the 2 video scenes changed -> LB delta x 7/2 = per-video-scene aa gain.
EOF
echo "=== ROUND14 BUILT: sub_round14_videoaa.zip ==="

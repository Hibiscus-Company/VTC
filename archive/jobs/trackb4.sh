#!/bin/bash
# GPU1 queue: exp22 warm-start finetune -> exp23 affine-interp (idea 1c)
#             -> exp24 bilateral grid (idea 1b)
# Base for 23/24 = exp19/20 config (opacity_reg 0.01 stock, noise_stop 25k);
# exp21's opacity_reg 0.002 REJECTED (74.28, overfit).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh
CHAMP_PLY=output/HCM0181_e10gate1/point_cloud/iteration_30000/point_cloud.ply
[ -f "$CHAMP_PLY" ] || { echo "ERROR champ ply missing: $CHAMP_PLY"; exit 1; }

run_b() {  # name, train-args...
  local NAME=$1; shift
  local OUT=/mnt/d/avv/output/HCM0181_$NAME
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
    --source $S/train --out $OUT "$@" || { echo "=== $NAME TRAIN FAILED ==="; return 1; }
  echo "=== $NAME TRAIN DONE ==="
  local APP=""
  [ -f "$OUT/app_affine.pt" ] && APP="--app_affine $OUT/app_affine.pt"
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    --distort auto --sparse $S/train/sparse/0 $APP 2>&1 | tail -2
  conda activate fastgs2
  mkdir -p ~/densq/$NAME
  ln -sfn $OUT/test_poses_renders ~/densq/$NAME/HCM0181
  CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$NAME --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
  echo "=== $NAME SCORED ==="
}

# exp22: warm-start from FastGS champion, 15k finetune, 5k lpips tail
run_b gsplatB4warm --init_ply "$CHAMP_PLY" --iters 15000 --cap_max 5500000 \
  --refine_stop 10000 --noise_stop 10000 --lpips_from 10000

# exp23: idea 1c per-image affine + trajectory interpolation (base exp20 cfg)
run_b gsplatB5affine --iters 30000 --cap_max 5000000 --noise_stop 25000 --app_affine

# exp24: idea 1b bilateral grid (base exp20 cfg)
run_b gsplatB6bilagrid --iters 30000 --cap_max 5000000 --noise_stop 25000 --bilagrid

echo "TRACKB4 QUEUE DONE"

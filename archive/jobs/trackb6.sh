#!/bin/bash
# exp26 = audit round-7 item-3 pick: warm-start PURE finetune ("MCMC forces
# off"): no relocation (refine_stop 0), no noise (noise_stop 0), no
# opacity/scale reg, short run, early lpips, means-lr x0.3. Goal: keep the
# FastGS opacity structure intact and only add antialiased adaptation + LPIPS
# polish. Expected +0.15..0.35 over exp22 (74.82) -> ~75.0-75.2, and the best
# decorrelated ensemble member.
# Chains after renderA (GPU1: exp25b -> renderA -> exp26).
set -o pipefail
until grep -q "MEMBER A PNG DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/renderA.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh
NAME=gsplatB8pure
OUT=/mnt/d/avv/output/HCM0181_$NAME
CHAMP_PLY=output/HCM0181_e10gate1/point_cloud/iteration_30000/point_cloud.ply

conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $S/train --out $OUT --init_ply "$CHAMP_PLY" \
  --iters 10000 --cap_max 5500000 --refine_stop 0 --noise_stop 0 \
  --opacity_reg 0 --scale_reg 0 --lpips_from 3000 --means_lr 4.8e-5 \
  || { echo "=== $NAME TRAIN FAILED ==="; exit 1; }
echo "=== $NAME TRAIN DONE ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
  --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
  --distort auto --sparse $S/train/sparse/0 2>&1 | tail -2
conda activate fastgs2
mkdir -p ~/densq/$NAME
ln -sfn $OUT/test_poses_renders ~/densq/$NAME/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$NAME --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
echo "=== $NAME SCORED ==="
echo "TRACKB6 QUEUE DONE"

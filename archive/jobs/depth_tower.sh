#!/bin/bash
# Bucket-C depth-prior on a TOWER (HCM0181 proxy -- proper eval-split + cached baselines:
# ema999 solo 75.4293, ensemble mean 76.3883). Tests the "clean-SfM-so-no-help" assumption
# directly instead of assuming it. Real tower geometry weakness = open-sky ~0-SfM floaters,
# exactly what a depth prior regularizes. Gated on chair depth run freeing GPU1 (no stacking).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/HCM0181
M=/mnt/d/avv/depth/HCM0181_dp05

while [ ! -f /mnt/d/avv/depth_chair.DONE ]; do sleep 60; done
echo "=== GPU1 free -- tower depth-prior (HCM0181) $(date) ==="
conda activate gsplat
# precompute Depth-Anything for tower train imgs (fast, light)
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_depth.py \
  --images $ES/train_sub/images --out /mnt/d/avv/depth_prior/HCM0181 --device cuda:0 2>&1 | tail -1
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $ES/train_sub --images images --seed 42 --ut --out $M \
  --depth_prior 0.05 --depth_dir /mnt/d/avv/depth_prior/HCM0181 \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 \
  2>&1 | tail -3 || { echo "!!! DEPTH TOWER TRAIN FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
conda activate fastgs2
echo "=== SCORES (tower ema999 solo 75.4293) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag tower_depth_solo
echo "=== ensemble-add (ref mean 76.3883) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/combiner_sweep.py --tag tower_depthadd --out_root /mnt/d/avv/depth \
  --dirs /mnt/d/avv/tw_test/HCM0181_ema999/eval_png /mnt/d/avv/tw_test/HCM0181_ema099/eval_png \
         /mnt/d/avv/tw_test/HCM0181_minop02/eval_png /mnt/d/avv/tw_test/HCM0181_skydome50k/eval_png \
         $M/eval_png \
  --gt_dir $ES/eval_gt 2>/dev/null | grep " mean "
echo "=== DEPTH TOWER DONE $(date) ==="
touch /mnt/d/avv/depth_tower.DONE

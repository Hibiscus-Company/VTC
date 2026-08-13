#!/bin/bash
# Bucket-C: monocular depth-prior (MICCAI endoscopic-GS technique) on chair eval-split.
# Full chair recipe + --depth_prior 0.05, comparable to chair_ema099 eval baseline 69.8051.
# Decision protocol: score solo vs baseline AND as an added member to the r20 chair eval ensemble.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/chair
M=/mnt/d/avv/depth/chair_dp05
conda activate gsplat
echo "=== chair depth-prior 0.05 training $(date) ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $ES/train_sub --images images --seed 42 --out $M \
  --depth_prior 0.05 --depth_dir /mnt/d/avv/depth_prior/chair \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000 \
  2>&1 | tail -3 || { echo "!!! DEPTH CHAIR TRAIN FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
conda activate fastgs2
echo "=== SCORES (chair baseline eval 69.8051) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/eval_score.py --render_dir $M/eval_png \
  --gt_dir $ES/eval_gt --tag chair_depth_solo
echo "=== ensemble-add: depth member joins the chair eval ensemble (ref mean 71.0682) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/combiner_sweep.py --tag chair_depthadd --out_root /mnt/d/avv/depth \
  --dirs /mnt/d/avv/tw_test/chair_ema099/eval_png /mnt/d/avv/tw_test/chair_ema999/eval_png \
         /mnt/d/avv/tw_test/chair_capmax2M/eval_png /mnt/d/avv/tw_test/chair_aniso01/eval_png \
         $M/eval_png \
  --gt_dir $ES/eval_gt 2>/dev/null | grep " mean "
echo "=== DEPTH CHAIR DONE $(date) ==="
touch /mnt/d/avv/depth_chair.DONE

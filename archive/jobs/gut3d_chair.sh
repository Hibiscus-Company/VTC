#!/bin/bash
# MODULE TEST (user's Q): extend 3DGUT's with_eval3d to VIDEO. --ut on pinhole chair = 3D-accurate
# Gaussian eval (unscented) with zero distortion (k1=0) -> a decorrelated video member vs the
# 2D-EWA antialiased baseline. GPU1 (freed by bonsai-2dgs crash). Eval-split, solo + ensemble-add.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/chair; M=/mnt/d/avv/gut3d/chair
conda activate gsplat
echo "=== 3DGUT-eval3d video (chair, --ut k1=0) $(date) ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed 42 --ut --out $M \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000 \
  2>&1 | tail -3 || { echo "!!! GUT3D CHAIR FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
conda activate fastgs2
echo "=== chair 3DGUT-video scores (baseline solo 69.8051 / ens-mean 71.0682) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag chair_gut3d_solo
CUDA_VISIBLE_DEVICES=1 python scripts/combiner_sweep.py --tag chair_gut3dadd --out_root /mnt/d/avv/gut3d \
  --dirs /mnt/d/avv/tw_test/chair_ema099/eval_png /mnt/d/avv/tw_test/chair_ema999/eval_png \
         /mnt/d/avv/tw_test/chair_capmax2M/eval_png /mnt/d/avv/tw_test/chair_aniso01/eval_png \
         $M/eval_png --gt_dir $ES/eval_gt 2>/dev/null | grep " mean "
echo "=== GUT3D CHAIR DONE $(date) ==="
touch /mnt/d/avv/gut3d_chair.DONE

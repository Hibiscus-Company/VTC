#!/bin/bash
# Paired arm to tower_ema_production.sh (GPU1, HCM0421 seed7 decay0.999, running since 04:33).
# Screen-tier just showed a SIGN-FLIP vs the full-length proxy: full-length HCM0181 proxy had
# decay0.999 (+0.484) beat decay0.99 (+0.366); screen-tier HCM0181 proxy has decay0.99 (+0.28)
# beat decay0.999 (+0.10). Can't trust either proxy alone for the REAL tower's decay choice --
# running BOTH decays on the SAME real scene+seed (HCM0421, seed7) for a direct, clean answer.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
S=HCM0421
M=/mnt/d/avv/r17/${S}_ut7_ema099
TOW_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

conda activate gsplat
echo "=== TOWER EMA PRODUCTION [$S] seed7 decay0.99 starting $(date) ==="
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
  --source $PRI/$S/train --images images --ut --seed 7 --ema_decay 0.99 \
  --out $M $TOW_ARGS 2>&1 | tail -3 || { echo "!!! TOWER EMA 099 PRODUCTION TRAIN FAIL"; exit 1; }
echo "--ut --seed 7 --ema_decay 0.99 $TOW_ARGS" > $M/train_args.txt

CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
  --ckpt $M/ckpt.pt --csv $PRI/$S/test/test_poses.csv \
  --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
echo "=== TOWER EMA 099 PRODUCTION [$S] DONE $(date), $(ls $M/test_png | wc -l) test renders ready ==="
touch /mnt/d/avv/tower_ema_production_099.DONE

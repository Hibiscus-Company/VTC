#!/bin/bash
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONUNBUFFERED=1
PRI=/mnt/d/avv/data/phase1/private_set2; S=HCM0674; M=/mnt/d/avv/r17/${S}_ut7_ema999
conda activate gsplat
echo "=== HCM0674 EMA re-run (clean GPU1) $(date) ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py --source $PRI/$S/train --images images \
  --ut --seed 7 --ema_decay 0.999 --out $M \
  --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 \
  2>&1 | tail -2 || { echo "!!! HCM0674 REDO FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $PRI/$S/test/test_poses.csv --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
echo "=== HCM0674 EMA DONE $(date), $(ls $M/test_png|wc -l) renders ==="
touch /mnt/d/avv/hcm0674_redo.DONE

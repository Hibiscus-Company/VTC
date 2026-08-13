#!/bin/bash
# 2DGS feasibility probe (moonshot). Gated on GPU0 freeing (chair_depth_prod.DONE) -- no stacking.
# Stage 1: 200-iter smoke (catch integration crashes fast). Stage 2: full 30k eval-split run on
# chair -> render eval -> score solo (vs 69.8051) AND as ensemble-add (vs ref mean 71.0682).
# Decision: viable decorrelated member if solo is respectable (>~66) OR ensemble-add is positive.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/chair
conda activate gsplat

while [ ! -f /mnt/d/avv/chair_depth_prod.DONE ]; do sleep 60; done
echo "=== GPU0 free -- 2DGS SMOKE (200 iter) $(date) ==="
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_2dgs.py --source $ES/train_sub --images images \
  --out /mnt/d/avv/twodgs/smoke --iters 200 --refine_start 100 --refine_stop 150 --reset_every 500 \
  --dist_from 100 --normal_from 150 --lpips_from 100000 2>&1 | grep -aE "\[0\]|\[100\]|loss|Error|Traceback|assert|saved" | grep -vE "FutureWarning|Warning" | tail -8
if ! grep -aq "saved" <(tail -50 /home/bkai/.claude/jobs/1c9cf7e9/tmp/twodgs_probe.log 2>/dev/null); then
  echo "!!! 2DGS SMOKE did not reach save -- checking..."; fi

echo "=== 2DGS full eval-split run (chair, 30k) $(date) ==="
M=/mnt/d/avv/twodgs/chair
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_2dgs.py --source $ES/train_sub --images images \
  --out $M --iters 30000 --refine_stop 15000 --lpips_from 15000 \
  --csv $ES/eval_poses.csv --png_dir $M/eval_png 2>&1 | tail -4 || { echo "!!! 2DGS FULL FAIL"; exit 1; }
conda activate fastgs2
echo "=== SCORES (chair baseline 69.8051 | ref ens mean 71.0682) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag chair_2dgs_solo
CUDA_VISIBLE_DEVICES=0 python scripts/combiner_sweep.py --tag chair_2dgsadd --out_root /mnt/d/avv/twodgs \
  --dirs /mnt/d/avv/tw_test/chair_ema099/eval_png /mnt/d/avv/tw_test/chair_ema999/eval_png \
         /mnt/d/avv/tw_test/chair_capmax2M/eval_png /mnt/d/avv/tw_test/chair_aniso01/eval_png \
         $M/eval_png \
  --gt_dir $ES/eval_gt 2>/dev/null | grep " mean "
echo "=== 2DGS PROBE DONE $(date) ==="
touch /mnt/d/avv/twodgs_probe.DONE

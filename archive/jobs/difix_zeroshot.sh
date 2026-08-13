#!/bin/bash
# H-A ZERO-SHOT: Difix3D+ (released nvidia/difix) at 3 strengths on two existing render sets
# WITH local GT: (a) real tower HCM0421 in-sample 40 renders (baseline SCORE 82.0100),
# (b) chair eval-split 58 renders (baseline 69.8051). Decides whether the restoration class
# has signal here. Gated on the env-ready sentinel; needs the Difix3D repo for pipeline_difix.
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
while [ ! -f /mnt/d/avv/difix_env.READY ]; do sleep 60; done

cd /home/bkai/.claude/jobs/1c9cf7e9/tmp
if [ ! -d Difix3D ]; then git clone --depth 1 https://github.com/nv-tlabs/Difix3D.git || exit 1; fi
export PYTHONPATH=/home/bkai/.claude/jobs/1c9cf7e9/tmp/Difix3D/src:/home/bkai/.claude/jobs/1c9cf7e9/tmp/Difix3D
cd /mnt/c/Users/BKAI/an_plaza2/FastGS

conda activate difix
for S in 1.0 0.5 0.25; do
  CUDA_VISIBLE_DEVICES=0 python scripts/difix_run.py \
    --in_dir /mnt/d/avv/r17/HCM0421_ut7_ema999/insample_png \
    --out_dir /mnt/d/avv/difix/HCM0421_s${S} --strength $S 2>&1 | tail -1 || { echo "!!! DIFIX TOWER s$S FAIL"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 python scripts/difix_run.py \
    --in_dir /mnt/d/avv/tw_test/chair_ema099/eval_png \
    --out_dir /mnt/d/avv/difix/chair_s${S} --strength $S 2>&1 | tail -1 || { echo "!!! DIFIX CHAIR s$S FAIL"; exit 1; }
done

conda activate fastgs2
echo "=== SCORES: tower baseline 82.0100 | chair baseline 69.8051 ==="
for S in 1.0 0.5 0.25; do
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir /mnt/d/avv/difix/HCM0421_s${S} \
    --gt_dir /mnt/d/avv/evalsplit/HCM0421/eval_gt --tag "HCM0421_difix_s${S}"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir /mnt/d/avv/difix/chair_s${S} \
    --gt_dir /mnt/d/avv/evalsplit/chair/eval_gt --tag "chair_difix_s${S}"
done
echo "=== DIFIX ZERO-SHOT DONE $(date) ==="
touch /mnt/d/avv/difix_zeroshot.DONE

#!/bin/bash
# H-B bound test driver, chair. Uses the EXISTING eval-split chair_ema099 model (eval score
# 69.8051 baseline). Steps: build train_poses.csv -> render train_sub poses -> fit per-frame
# transform on train pairs -> interpolate to eval holes -> apply -> score (full arm + gain-only
# ablation to separate blur from exposure). GPU-light; runs opportunistically on GPU0 alongside
# the tower trainer (its own OOM would kill only this script, not the trainer).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/chair
M=/mnt/d/avv/tw_test/chair_ema099
OUT=/mnt/d/avv/blurbound/chair
mkdir -p $OUT

conda activate fastgs2
python - <<'PY'
import csv, os, sys
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/scripts")
from make_eval_split import qvec, read_cam
ES = "/mnt/d/avv/evalsplit/chair"
sp = os.path.join(ES, "train_sub", "sparse", "0")
names = sorted(os.listdir(os.path.join(ES, "train_sub", "images")))
fx, fy, cx, cy, w, h = read_cam(os.path.join(sp, "cameras.bin"))
poses = qvec(os.path.join(sp, "images.bin"), set(names))
missing = [n for n in names if n not in poses]
assert not missing, f"poses missing for {missing[:5]}"
with open("/mnt/d/avv/blurbound/chair/train_poses.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image_name","qw","qx","qy","qz","tx","ty","tz","fx","fy","cx","cy","width","height"])
    for n in names:
        q, t = poses[n]
        wr.writerow([n, *q, *t, fx, fy, cx, cy, w, h])
print(f"train_poses.csv: {len(names)} rows")
PY

conda activate gsplat
CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
  --ckpt $M/ckpt.pt --csv $OUT/train_poses.csv \
  --out $OUT/train_render --png_dir $OUT/train_png 2>&1 | tail -1 \
  || { echo "!!! TRAIN RENDER FAIL (likely OOM on busy GPU -- rerun when free)"; exit 1; }

conda activate fastgs2
CUDA_VISIBLE_DEVICES=0 python scripts/blur_bound.py \
  --train_render $OUT/train_png --train_gt $ES/train_sub/images \
  --eval_render $M/eval_png --out $OUT/eval_corrected \
  --params_json $OUT/fits.json 2>&1 | tail -8

CUDA_VISIBLE_DEVICES=0 python scripts/blur_bound.py \
  --train_render $OUT/train_png --train_gt $ES/train_sub/images \
  --eval_render $M/eval_png --out $OUT/eval_gainonly \
  --params_json $OUT/fits_gainonly.json --gain_only 2>&1 | tail -4

echo "=== SCORES (baseline chair_ema099 = 69.8051) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir $OUT/eval_corrected \
  --gt_dir $ES/eval_gt --tag chair_blurmatch
CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir $OUT/eval_gainonly \
  --gt_dir $ES/eval_gt --tag chair_gainonly
echo "=== BLUR BOUND CHAIR DONE $(date) ==="
touch /mnt/d/avv/blurbound_chair.DONE

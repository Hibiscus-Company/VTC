#!/bin/bash
# H-B bound test, BONSAI -- the scene the strategy audit flagged hardest (VoL p5-median
# 126-474 = 4x within-scene blur spread). Chair's bound came back weak (+0.094); bonsai is
# H-B's last stand. Same protocol: train renders from the existing eval-split ckpt
# (tw_test/bonsai_ema099, eval 71.8829 ~ baseline 71.90), per-frame fit, frame-index interp.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai
M=/mnt/d/avv/tw_test/bonsai_ema099
OUT=/mnt/d/avv/blurbound/bonsai
mkdir -p $OUT

conda activate fastgs2
python - <<'PY'
import csv, os, sys
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/scripts")
from make_eval_split import qvec, read_cam
ES = "/mnt/d/avv/evalsplit/bonsai"
sp = os.path.join(ES, "train_sub", "sparse", "0")
names = sorted(os.listdir(os.path.join(ES, "train_sub", "images")))
fx, fy, cx, cy, w, h = read_cam(os.path.join(sp, "cameras.bin"))
poses = qvec(os.path.join(sp, "images.bin"), set(names))
names = [n for n in names if n in poses]
with open("/mnt/d/avv/blurbound/bonsai/train_poses.csv", "w", newline="") as f:
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
  || { echo "!!! TRAIN RENDER FAIL"; exit 1; }

conda activate fastgs2
CUDA_VISIBLE_DEVICES=0 python scripts/blur_bound.py \
  --train_render $OUT/train_png --train_gt $ES/train_sub/images \
  --eval_render $M/eval_png --out $OUT/eval_corrected \
  --params_json $OUT/fits.json 2>&1 | tail -6

echo "=== SCORE (baseline bonsai_ema099 = 71.8829) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py --render_dir $OUT/eval_corrected \
  --gt_dir $ES/eval_gt --tag bonsai_blurmatch
echo "=== BLUR BOUND BONSAI DONE $(date) ==="
touch /mnt/d/avv/blurbound_bonsai.DONE

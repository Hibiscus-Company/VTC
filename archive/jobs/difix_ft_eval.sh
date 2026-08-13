#!/bin/bash
# Difix FT go/no-go: run the fine-tuned checkpoint over chair's 58 eval-hole renders at native
# res (720x1280, both /8), score vs GT. Kill bar: must beat baseline 69.8051 by >= +0.3.
# Gated on training-done sentinel.
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
D3=/home/bkai/.claude/jobs/1c9cf7e9/tmp/Difix3D
export PYTHONPATH=$D3/src
IN=/mnt/d/avv/tw_test/chair_ema099/eval_png
GT=/mnt/d/avv/evalsplit/chair/eval_gt

while [ ! -f /mnt/d/avv/difix_ft_chair_trained.DONE ]; do sleep 30; done
CKPT=$(ls -t /mnt/d/avv/difix_ft/chair/run/checkpoints/model_*.pkl 2>/dev/null | head -1)
echo "=== Difix FT eval, ckpt=$CKPT $(date) ==="

conda activate difix
for STR in 1.0 0.5; do
  OUT=/mnt/d/avv/difix_ft/chair/eval_ft_s${STR}
  CUDA_VISIBLE_DEVICES=0 python - "$CKPT" "$IN" "$OUT" "$STR" <<'PY'
import sys, os, glob
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/Difix3D/src")
import numpy as np
from PIL import Image
import torch
from model import Difix
ckpt, indir, outdir, strength = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
os.makedirs(outdir, exist_ok=True)
m = Difix(pretrained_path=ckpt, timestep=199, mv_unet=False); m.set_eval()
files = sorted(glob.glob(os.path.join(indir, "*.png")))
for f in files:
    img = Image.open(f).convert("RGB"); W, H = img.size
    out = m.sample(img, height=H, width=W, ref_image=None, prompt="remove degradation")
    out = out.resize((W, H))
    if strength < 1.0:
        a = np.asarray(img, np.float32); b = np.asarray(out, np.float32)
        out = Image.fromarray(np.clip((1-strength)*a + strength*b + 0.5, 0, 255).astype(np.uint8))
    out.save(os.path.join(outdir, os.path.basename(f)))
print("wrote", len(files), "to", outdir)
PY
done

conda activate fastgs2
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
echo "=== SCORES (chair baseline = 69.8051; kill bar +0.3 -> 70.11) ==="
for STR in 1.0 0.5; do
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py \
    --render_dir /mnt/d/avv/difix_ft/chair/eval_ft_s${STR} --gt_dir $GT --tag "chair_ft_s${STR}"
done
echo "=== DIFIX FT EVAL DONE $(date) ==="
touch /mnt/d/avv/difix_ft_eval.DONE

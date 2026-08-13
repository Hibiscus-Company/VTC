#!/bin/bash
# Per-pose finetune A/B smoke (task #13, audit round-8 rank-2).
# Chains after IBR AB on GPU0. First pass: 12 poses only (~15 min) to gauge
# the effect size before committing an hour to full-60; scored on the
# 12-image subset vs the SAME subset of the splat-only baseline.
set -o pipefail
until grep -q "IBR AB DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/ibr_ab.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
CKPT=/mnt/d/avv/output/HCM0181_gsplatB10ut8M/ckpt.pt
OUT=/mnt/d/avv/ibr/HCM0181_ppft12
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
CUDA_VISIBLE_DEVICES=0 python gsplat_track/perpose_finetune.py \
  --ckpt $CKPT --source $S/train --csv $S/test/test_poses.csv \
  --out $OUT/renders --png_dir $OUT/renders_png --limit 12 --stride 5 \
  2>&1 | grep -vE "done$" | tail -3 \
  || { echo "=== PPFT FAILED ==="; exit 1; }
echo "=== PPFT 12-pose RENDER DONE ==="
# subset-vs-subset comparison against the exp28 baseline renders
conda activate fastgs2
python - <<'EOF'
import os, shutil
src = "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders"
sub = os.path.expanduser("~/densq/ppft_base12/HCM0181")
os.makedirs(sub, exist_ok=True)
done = sorted(os.listdir("/mnt/d/avv/ibr/HCM0181_ppft12/renders"))
for n in done:
    shutil.copy2(os.path.join(src, n), os.path.join(sub, n))
print(f"baseline subset: {len(done)} images")
EOF
mkdir -p ~/densq/ppft12
ln -sfn $OUT/renders ~/densq/ppft12/HCM0181
CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub ~/densq/ppft_base12 --device cuda:0 2>&1 | grep -E "Score\(vgg\)" | sed 's/^/PPFT base12 /'
CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub ~/densq/ppft12 --device cuda:0 2>&1 | grep -E "Score\(vgg\)" | sed 's/^/PPFT tuned12 /'
echo "PPFT AB DONE"

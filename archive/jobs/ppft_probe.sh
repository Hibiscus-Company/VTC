#!/bin/bash
# Fast per-pose finetune PROBE (clean GPU0): does the DIRECTION help at all?
# color_only (freeze geometry, ~5x faster, no floater risk) + no LPIPS in loop
# + 200 steps + 6 poses stride-10. Goal: effect-size signal in ~10 min to
# decide whether per-pose is worth a production-efficient redesign.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
CKPT=/mnt/d/avv/output/HCM0181_gsplatB10ut8M/ckpt.pt
OUT=/mnt/d/avv/ibr/HCM0181_ppftprobe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "PROBE_WALL %e s" \
python gsplat_track/perpose_finetune.py \
  --ckpt $CKPT --source $S/train --csv $S/test/test_poses.csv \
  --out $OUT/renders --png_dir $OUT/renders_png \
  --color_only --lambda_lpips 0 --steps 200 --k 5 --limit 6 --stride 10 \
  2>&1 | grep -E "PROBE_WALL|gaussians|done$|Wrote|Error|Traceback" | tail -10 \
  || { echo "=== PROBE FAILED ==="; exit 1; }
echo "=== PPFT PROBE RENDERED ==="
conda activate fastgs2
# subset baseline from exp28 renders
python - <<'EOF'
import os, shutil
src="/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders"
sub=os.path.expanduser("~/densq/ppftprobe_base/HCM0181"); os.makedirs(sub, exist_ok=True)
for n in sorted(os.listdir("/mnt/d/avv/ibr/HCM0181_ppftprobe/renders")):
    shutil.copy2(os.path.join(src,n), os.path.join(sub,n))
print("baseline subset:", len(os.listdir(sub)))
EOF
mkdir -p ~/densq/ppftprobe
ln -sfn $OUT/renders ~/densq/ppftprobe/HCM0181
CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub ~/densq/ppftprobe_base --device cuda:0 2>&1 | grep -E "Score\(vgg\)" | sed 's/^/PROBE base /'
CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub ~/densq/ppftprobe --device cuda:0 2>&1 | grep -E "Score\(vgg\)" | sed 's/^/PROBE tuned /'
echo "PPFT PROBE DONE"

#!/bin/bash
# exp25b = idea 1a PPISP, CORRECTED per audit round 7: controller activation
# moved 24000->29000 (--ppisp_activation 29/30) and lpips_from 25000->24000 so
# the splats get a live-radiance LPIPS phase (24k-29k) before the 1k
# distillation window (during which rgb.detach() cuts photometric grads and
# train_gsplat.py now zeroes opacity/scale reg). exp25 (activation 24k) was
# killed at ~step 8k: its whole lpips phase would have trained only the
# controller while reg drained opacity unopposed.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
source ~/miniconda3/etc/profile.d/conda.sh
NAME=gsplatB7ppisp2
OUT=/mnt/d/avv/output/HCM0181_$NAME

conda activate gsplat
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
  --source $S/train --out $OUT --iters 30000 --cap_max 5000000 \
  --noise_stop 24000 --lpips_from 24000 --ppisp --ppisp_activation 0.966667 \
  || { echo "=== $NAME TRAIN FAILED ==="; exit 1; }
echo "=== $NAME TRAIN DONE ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/ckpt.pt --csv $S/test/test_poses.csv \
  --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
  --distort auto --sparse $S/train/sparse/0 --ppisp $OUT/ppisp.pt 2>&1 | tail -2
conda activate fastgs2
mkdir -p ~/densq/$NAME
ln -sfn $OUT/test_poses_renders ~/densq/$NAME/HCM0181
CUDA_VISIBLE_DEVICES=1 python score_submission.py --sub ~/densq/$NAME --device cuda:0 2>&1 | grep -E "HCM0181 |Score\("
echo "=== $NAME SCORED ==="
echo "TRACKB5 QUEUE DONE"

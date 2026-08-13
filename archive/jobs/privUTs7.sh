#!/bin/bash
# Private seed-7 UT members (round-6 composition m6ut60 needs a 2nd UT seed
# per scene; proven +0.27 on HCM0181). Recipe: 30k/8M, --seed 7. GPU1.
# All 8 scenes here; if GPU0 frees early (post-IBR), split manually.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
for s in HNI0131 HNI0265 HNI0366 HNI0437 HCM0249 HCM0254 HCM0276 HCM1439; do
  OUT=/mnt/d/avv/output/${s}_gsplatUTs7
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
    --source $PRI/$s/train --images images --ut --seed 7 \
    --out $OUT --iters 30000 --cap_max 8000000 --noise_stop 25000 \
    || { echo "=== privUTs7 $s TRAIN FAILED ==="; continue; }
  echo "=== privUTs7 $s TRAIN DONE ==="
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py \
    --ckpt $OUT/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
    --out $OUT/test_poses_renders --png_dir $OUT/test_poses_renders_png \
    --ut_render native 2>&1 | tail -1
  echo "=== privUTs7 $s RENDERED ==="
done
echo "PRIVUTS7 DONE"

#!/bin/bash
# Member A (champ) private re-render with PNG dual-write (originals were
# pre-dual-write JPEG-only). GPU1, chained after exp25 (trackb5).
# Renders go to D: per storage rule; models stay on C: repo output/.
set -o pipefail
until grep -q "TRACKB5 QUEUE DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/trackb5.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  OUT=/mnt/d/avv/output/${s}_champA
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=1 python render_test_poses.py -m output/${s}_champ \
    --csv $PRI/$s/test/test_poses.csv --out $OUT/test_poses_renders \
    --png_dir $OUT/test_poses_renders_png --mult 0.7 \
    --distort auto --sparse $PRI/$s/train/sparse/0 \
    || { echo "=== renderA $s FAILED ==="; exit 1; }
  echo "=== renderA $s done ==="
done
echo "MEMBER A PNG DONE"

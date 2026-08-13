#!/bin/bash
# 2-member private ensemble (A=champ gate1, B=gate5) -> submission zip.
# Fires when member A PNG re-render and member B training are both done.
# CPU-only (PIL/numpy mean + JPEG quality ladder).
set -o pipefail
until grep -q "MEMBER A PNG DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/renderA.log 2>/dev/null \
   && grep -q "MEMBER B DONE"    /home/bkai/.claude/jobs/1c9cf7e9/tmp/queue15.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2

PAIRS=""
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  ENS=/mnt/d/avv/ens/private_AB/$s
  python ensemble_renders.py \
    --dirs /mnt/d/avv/output/${s}_champA/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memB/test_poses_renders_png \
    --out $ENS/jpg --png_dir $ENS/png \
    --names_from $PRI/$s/test/test_poses.csv \
    || { echo "=== ensemble $s FAILED ==="; exit 1; }
  echo "=== ensemble $s done ==="
  PAIRS="$PAIRS $s=$ENS/png"
done

python build_submission_zip.py --scene_dirs $PAIRS \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round2_ens2AB_private.zip \
  || { echo "=== ZIP BUILD FAILED ==="; exit 1; }
echo "ENS2 ZIP DONE"

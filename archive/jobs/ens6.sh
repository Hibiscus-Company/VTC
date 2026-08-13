#!/bin/bash
# R6 zip: 3 gates + 2 UT seeds (UT 30k/5M s42 + UTs7 30k/8M), family-weighted
# w_UT = 0.6 (gates 0.4/3 = 0.1333 each, UTs 0.3 each).
# Honest expectation: +0.1 over R5 (the private UT s42 member is still the weak
# 30k/5M recipe; R7 with 60k/8M members is the real upgrade).
set -o pipefail
until grep -q "PRIVUTS7 DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/privUTs7.log 2>/dev/null; do sleep 60; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2

PAIRS=""
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  ENS=/mnt/d/avv/ens/private_r6/$s
  python ensemble_renders.py \
    --dirs /mnt/d/avv/output/${s}_champA/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memB/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memC/test_poses_renders_png \
           /mnt/d/avv/output/${s}_gsplatB9ut/test_poses_renders_png \
           /mnt/d/avv/output/${s}_gsplatUTs7/test_poses_renders_png \
    --weights 0.1333 0.1333 0.1333 0.3 0.3 \
    --out $ENS/jpg --png_dir $ENS/png \
    --names_from $PRI/$s/test/test_poses.csv \
    || { echo "=== ens6 $s FAILED ==="; exit 1; }
  echo "=== ens6 $s done ==="
  PAIRS="$PAIRS $s=$ENS/png"
done

python build_submission_zip.py --scene_dirs $PAIRS \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round6_ens5w_private.zip \
  || { echo "=== ENS6 ZIP BUILD FAILED ==="; exit 1; }
echo "ENS6 ZIP DONE"

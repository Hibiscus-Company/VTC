#!/bin/bash
# Round-5 private ensemble: A(gate1) + B(gate5) + C(gate2) + UT(gsplatB9ut)
# per scene, FAMILY-WEIGHTED per audit round-9 sweep: single-UT peak at
# w_UT=0.5 (76.93 vs 76.48 uniform on HCM0181) — gates split 0.5, UT 0.5.
# memD (gate0) dropped (3h/scene, ~+0.1). Fires when both privUT halves done.
set -o pipefail
until grep -q "PRIVUT GPU0 DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/privUT_gpu0.log 2>/dev/null \
   && grep -q "PRIVUT GPU1 DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/privUT_gpu1.log 2>/dev/null; do sleep 180; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2

PAIRS=""
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  ENS=/mnt/d/avv/ens/private_m5ut/$s
  python ensemble_renders.py \
    --dirs /mnt/d/avv/output/${s}_champA/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memB/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memC/test_poses_renders_png \
           /mnt/d/avv/output/${s}_gsplatB9ut/test_poses_renders_png \
    --weights 0.1667 0.1667 0.1667 0.5 \
    --out $ENS/jpg --png_dir $ENS/png \
    --names_from $PRI/$s/test/test_poses.csv \
    || { echo "=== ens5 $s FAILED ==="; exit 1; }
  echo "=== ens5 $s done ==="
  PAIRS="$PAIRS $s=$ENS/png"
done

python build_submission_zip.py --scene_dirs $PAIRS \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round5_ensABCUT_private.zip \
  || { echo "=== ENS5 ZIP BUILD FAILED ==="; exit 1; }
echo "ENS5 ZIP DONE"

#!/bin/bash
# 3-member private ensemble (A=champ gate1, B=gate5, C=gate2) -> round-3 zip.
# Fires when memC finishes (queueCD.log). Round-3 projection: mean3 ≈ +0.65
# over single champ (vs +0.5 for the shipped ens2AB zip).
set -o pipefail
until grep -q "MEMBER C DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/queueCD.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2

PAIRS=""
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  ENS=/mnt/d/avv/ens/private_ABC/$s
  python ensemble_renders.py \
    --dirs /mnt/d/avv/output/${s}_champA/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memB/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memC/test_poses_renders_png \
    --out $ENS/jpg --png_dir $ENS/png \
    --names_from $PRI/$s/test/test_poses.csv \
    || { echo "=== ens3 $s FAILED ==="; exit 1; }
  echo "=== ens3 $s done ==="
  PAIRS="$PAIRS $s=$ENS/png"
done

python build_submission_zip.py --scene_dirs $PAIRS \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round3_ens3ABC_private.zip \
  || { echo "=== ENS3 ZIP BUILD FAILED ==="; exit 1; }
echo "ENS3 ZIP DONE"

#!/bin/bash
# 4-member private ensemble (A gate1, B gate5, C gate2, D gate0) -> round-4 zip.
# Fires when memD finishes (queueCD.log). Public curve: mean4 = best FastGS-only
# composition (75.980 on HCM0181); private realization measured 3.6x projection
# (ens2AB: projected +0.5, LB realized +1.82).
set -o pipefail
until grep -q "MEMBER D DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/queueCD.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
PRI=~/data/phase1/private_set1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2

PAIRS=""
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  ENS=/mnt/d/avv/ens/private_ABCD/$s
  python ensemble_renders.py \
    --dirs /mnt/d/avv/output/${s}_champA/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memB/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memC/test_poses_renders_png \
           /mnt/d/avv/output/${s}_memD/test_poses_renders_png \
    --out $ENS/jpg --png_dir $ENS/png \
    --names_from $PRI/$s/test/test_poses.csv \
    || { echo "=== ens4 $s FAILED ==="; exit 1; }
  echo "=== ens4 $s done ==="
  PAIRS="$PAIRS $s=$ENS/png"
done

python build_submission_zip.py --scene_dirs $PAIRS \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round4_ens4ABCD_private.zip \
  || { echo "=== ENS4 ZIP BUILD FAILED ==="; exit 1; }
echo "ENS4 ZIP DONE"

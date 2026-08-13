#!/bin/bash
# Deadband fix for the 4 towers not yet done (HCM0421 already in /mnt/d/avv/tmp_db_test).
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export OMP_NUM_THREADS=4
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
OUT=/mnt/d/avv/r33/tower_ens; FLD=/mnt/d/avv/fields_median_g1_g130
mkdir -p $OUT
for T in HCM0539 HCM0540 HCM0644 HCM0674; do
  mkdir -p $OUT/$T
  echo "=== $T restore(float mean, lam=1.0) ==="
  python $ER --mode apply --k 8 --lam 1.0 \
    --ens_dir /mnt/d/avv/r29/tower_ens/$T/png_ens \
    --member_dirs /mnt/d/avv/r25_mip3d/$T/test_png /mnt/d/avv/r28_members/$T/test_png \
    --mean_from_dirs /mnt/d/avv/r22/tower_ens/$T/png_ens /mnt/d/avv/r25_mip3d/$T/test_png /mnt/d/avv/r28_members/$T/test_png \
    --mean_weights 4 1 1 --out_dir $OUT/$T/png_er
  echo "=== $T field ==="
  python gsplat_track/apply_field.py --in_dir $OUT/$T/png_er --field $FLD/$T.npy \
    --out_dir $OUT/$T/png --strict
  n=$(ls $OUT/$T/png/*.png | wc -l); [ "$n" -eq 60 ] || { echo "!!! $T $n"; exit 1; }
  echo "<<< $T done ($n)"
done
# HCM0421 was built during the smoke test -- move it into place
mkdir -p $OUT/HCM0421
cp /mnt/d/avv/tmp_db_test/HCM0421_f/*.png $OUT/HCM0421/ 2>/dev/null || true
ls $OUT/HCM0421/*.png | wc -l
echo "=== ALL 5 TOWERS DEADBAND-FIXED ==="
touch /mnt/d/avv/towers_db.DONE

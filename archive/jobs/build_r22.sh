#!/bin/bash
# Build r22 = r21 + seed-101 as a new equal-weight tower member, all 5 set2 towers.
# Composition math (ensemble_renders.py normalizes weights internally, so passing the
# EXISTING member count N against 1 for the new member reproduces a true (N+1)-way
# equal mean: (N*mean_of_N + 1*new)/(N+1) == (sum_of_N + new)/(N+1)). Per r20/r21
# PROVENANCE.txt: HCM0421 already has 6 members (w 6:1 -> 7-way); HCM0539/0540/0644/0674
# have 5 (w 5:1 -> 6-way). Chair/bonsai untouched (byte-identical to r21).
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
conda activate fastgs2
OUT=/mnt/d/avv/r22
mkdir -p $OUT/tower_ens
DATA=/mnt/d/avv/data/phase1/private_set2

declare -A BASE=( [HCM0421]=/mnt/d/avv/r20/tower_ens/HCM0421/png_ens
                   [HCM0539]=/mnt/d/avv/r20/tower_ens/HCM0539/png_ens
                   [HCM0540]=/mnt/d/avv/r20/tower_ens/HCM0540/png_ens
                   [HCM0644]=/mnt/d/avv/r20/tower_ens/HCM0644/png_ens
                   [HCM0674]=/mnt/d/avv/r21/tower_ens/HCM0674/png_ens )
declare -A NBASE=( [HCM0421]=6 [HCM0539]=5 [HCM0540]=5 [HCM0644]=5 [HCM0674]=5 )

for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  echo "=== $T: ${NBASE[$T]}-member base + seed101 -> $((${NBASE[$T]}+1))-way equal mean ==="
  mkdir -p $OUT/tower_ens/$T
  python ensemble_renders.py --dirs "${BASE[$T]}" /mnt/d/avv/r22_seed101/$T/test_png \
    --weights ${NBASE[$T]} 1 \
    --out $OUT/tower_ens/$T/jpg_unfielded --png_dir $OUT/tower_ens/$T/png_ens \
    --names_from $DATA/$T/test/test_poses.csv
  python gsplat_track/apply_field.py --in_dir $OUT/tower_ens/$T/png_ens \
    --field /mnt/d/avv/r2r9/fields/$T.npy --out_dir $OUT/tower_ens/$T/png --strict
  n=$(ls $OUT/tower_ens/$T/png | grep -c '\.png$')
  echo "$T: $n field-corrected pngs ready"
done
echo "=== BUILD_R22 TOWER STAGE DONE $(date) ==="
touch /mnt/d/avv/build_r22_towers.DONE

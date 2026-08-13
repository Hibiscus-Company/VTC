#!/bin/bash
# waits for round14_build.sh to exit, then rebuilds round14 on R12 TOWERS (gates regressed
# -0.835/tower on LB; r13 composition must not ship). The chain's own zip is superseded.
set -uo pipefail
while pgrep -f "round14_build.sh" >/dev/null; do sleep 60; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
# r12 towers = r2r9 ens (60k/8M 2-seed UT + field) — the 76.649-graded composition
python build_submission_zip.py --scene_dirs \
  HCM0421=/mnt/d/avv/r2r9/ens/HCM0421/png HCM0539=/mnt/d/avv/r2r9/ens/HCM0539/png \
  HCM0540=/mnt/d/avv/r2r9/ens/HCM0540/png HCM0644=/mnt/d/avv/r2r9/ens/HCM0644/png \
  HCM0674=/mnt/d/avv/r2r9/ens/HCM0674/png \
  chair=/mnt/d/avv/r14/chair_ens/png \
  bonsai=/mnt/d/avv/r14/bonsai_ens/png \
  --data_root $PRI --out /mnt/d/avv/submissions/sub_round14b_videoaa_r12towers.zip
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round14b_videoaa_r12towers.zip --data_root $PRI
cat > /mnt/d/avv/submissions/sub_round14b_videoaa_r12towers.PROVENANCE.txt <<'EOP'
sub_round14b_videoaa_r12towers.zip — THE round14 to submit:
  towers: SAME as round12/round11 (UT 60k/8M 2-seed + field). NOT r13 gates (LB -0.835/tower).
  chair:  ANTIALIASED no-UT 60k lpearly 2-seed (K3 eval 69.37 vs 68.54), no field
  bonsai: ANTIALIASED no-UT 30k capD+lpearly 2-seed (K1 eval 71.90 vs 71.36), no field
  vs r12 only the 2 video scenes changed -> LB delta x 7/2 = per-video aa gain.
  DO NOT SUBMIT sub_round14_videoaa.zip (built on the regressed r13 towers).
EOP
echo "=== ROUND14b BUILT (r12 towers + aa videos) ==="

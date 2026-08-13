#!/bin/bash
# ROUND15 = r14b, but towers become a 3-SEED UT ensemble (add ut13 to ut7+ut42).
#   Move T1 from the strategy audit: third decorrelated UT seed on all 5 towers,
#   equal weight 1/3, +0.05-0.15/tower via the proven 1/N mechanism. Ship as an
#   all-tower swap so LB delta x 7/5 = the exact per-tower gain.
# Towers: ensemble 3 seeds -> apply the SAME DIS field (fit on train, seed-independent).
# Video scenes: UNCHANGED from r14b (r14 aa ensembles), do not touch.
# NO set -u (conda hooks reference unbound vars); pipefail + explicit || checks.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
R9=/mnt/d/avv/r2r9
OUT=/mnt/d/avv/r15
mkdir -p $OUT

PAIRS=""
for s in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  E=$OUT/ens/$s
  # verify all 3 seed renders + the field exist before touching anything
  for sd in ut7 ut42 ut13; do
    n=$(ls $R9/models/${s}_$sd/test_png 2>/dev/null | wc -l)
    [[ $n -eq 60 ]] || { echo "!!! [$s] seed $sd has $n/60 pngs -- ABORT"; exit 1; }
  done
  [[ -f $R9/fields/$s.npy ]] || { echo "!!! [$s] missing field -- ABORT"; exit 1; }
  # 3-seed equal-weight pixel-mean (float32 accumulate, single round -- ensemble_renders rules)
  python ensemble_renders.py \
    --dirs $R9/models/${s}_ut7/test_png $R9/models/${s}_ut42/test_png $R9/models/${s}_ut13/test_png \
    --weights 0.333333 0.333333 0.333333 --masks none none none \
    --out $E/jpg --png_dir $E/png_ens \
    --names_from $PRI/$s/test/test_poses.csv || { echo "!!! [$s] ENSEMBLE FAILED"; exit 1; }
  # apply the SAME train-fit DIS lens field the 2-seed r12/r14b towers used
  python gsplat_track/apply_field.py --strict \
    --in_dir $E/png_ens --field $R9/fields/$s.npy --out_dir $E/png \
    || { echo "!!! [$s] APPLY_FIELD FAILED"; exit 1; }
  cnt=$(ls $E/png/*.png 2>/dev/null | wc -l)
  [[ $cnt -eq 60 ]] || { echo "!!! [$s] final png has $cnt/60 -- ABORT"; exit 1; }
  echo "=== [$s] 3-seed tower ready ($cnt imgs) ==="
  PAIRS="$PAIRS $s=$E/png"
done

# video scenes: reuse r14b's aa ensembles verbatim
PAIRS="$PAIRS chair=/mnt/d/avv/r14/chair_ens/png bonsai=/mnt/d/avv/r14/bonsai_ens/png"

ZIP=/mnt/d/avv/submissions/sub_round15_3seedtowers_videoaa.zip
python build_submission_zip.py --scene_dirs $PAIRS --data_root $PRI --out $ZIP \
  || { echo "!!! BUILD_ZIP FAILED"; exit 1; }
python scripts/verify_zip.py --zip $ZIP --data_root $PRI || { echo "!!! VERIFY FAILED"; exit 1; }

cat > ${ZIP%.zip}.PROVENANCE.txt <<'EOP'
sub_round15_3seedtowers_videoaa.zip -- round15 candidate (all-tower swap vs r14b):
  towers: UT 60k/8M 3-SEED ensemble (ut7+ut42+ut13, w=1/3 each) + DIS lens field.
          ut13 trained identical recipe (60k/8M refine50k noise50k lpips_from50k),
          rendered --ut_render native. Adds a 3rd decorrelated member (1/N gain).
  chair:  UNCHANGED from r14b -- ANTIALIASED no-UT 60k lpearly 2-seed, no field.
  bonsai: UNCHANGED from r14b -- ANTIALIASED no-UT 30k capD+lpearly 2-seed, no field.
  vs r14b ONLY the 5 towers changed -> LB delta x 7/5 = per-tower 3rd-seed gain.
  Baseline to beat: r14b = 76.78777.
EOP
echo "=== ROUND15 BUILT -> $ZIP ==="
ls -la $ZIP

#!/bin/bash
# ROUND16 = round15 extended one more seed on every scene, same proven 1/N mechanism:
#   towers: 3-seed -> 4-seed UT ensemble (ut7+ut42+ut13+ut77, w=1/4) + SAME DIS field
#   chair/bonsai: 2-seed -> 3-seed AA ensemble (aa42+aa7+aa13, w=1/3), no field (as r14b/r15)
# NO set -u (conda hooks reference unbound vars); pipefail + explicit || checks.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
R9=/mnt/d/avv/r2r9
OUT=/mnt/d/avv/r16
mkdir -p $OUT

PAIRS=""
# ---- towers: 4-seed ----
for s in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  E=$OUT/tower_ens/$s
  for sd in ut7 ut42 ut13 ut77; do
    n=$(ls $R9/models/${s}_$sd/test_png 2>/dev/null | wc -l)
    [[ $n -eq 60 ]] || { echo "!!! [$s] seed $sd has $n/60 pngs -- ABORT"; exit 1; }
  done
  [[ -f $R9/fields/$s.npy ]] || { echo "!!! [$s] missing field -- ABORT"; exit 1; }
  python ensemble_renders.py \
    --dirs $R9/models/${s}_ut7/test_png $R9/models/${s}_ut42/test_png \
           $R9/models/${s}_ut13/test_png $R9/models/${s}_ut77/test_png \
    --weights 0.25 0.25 0.25 0.25 --masks none none none none \
    --out $E/jpg --png_dir $E/png_ens \
    --names_from $PRI/$s/test/test_poses.csv || { echo "!!! [$s] ENSEMBLE FAILED"; exit 1; }
  python gsplat_track/apply_field.py --strict \
    --in_dir $E/png_ens --field $R9/fields/$s.npy --out_dir $E/png \
    || { echo "!!! [$s] APPLY_FIELD FAILED"; exit 1; }
  cnt=$(ls $E/png/*.png 2>/dev/null | wc -l)
  [[ $cnt -eq 60 ]] || { echo "!!! [$s] final png has $cnt/60 -- ABORT"; exit 1; }
  echo "=== [$s] 4-seed tower ready ($cnt imgs) ==="
  PAIRS="$PAIRS $s=$E/png"
done

# ---- video: 3-seed, no field ----
for sc in chair bonsai; do
  E=$OUT/video_ens/$sc
  N=58; [[ $sc == bonsai ]] && N=28
  for sd in aa42 aa7 aa13; do
    n=$(ls /mnt/d/avv/r14/${sc}_${sd}/test_png 2>/dev/null | wc -l)
    [[ $n -eq $N ]] || { echo "!!! [$sc] seed $sd has $n/$N pngs -- ABORT"; exit 1; }
  done
  python ensemble_renders.py \
    --dirs /mnt/d/avv/r14/${sc}_aa42/test_png /mnt/d/avv/r14/${sc}_aa7/test_png /mnt/d/avv/r14/${sc}_aa13/test_png \
    --weights 0.333333 0.333333 0.333333 --masks none none none \
    --out $E/jpg --png_dir $E/png \
    --names_from $PRI/$sc/test/test_poses.csv || { echo "!!! [$sc] ENSEMBLE FAILED"; exit 1; }
  cnt=$(ls $E/png/*.png 2>/dev/null | wc -l)
  [[ $cnt -eq $N ]] || { echo "!!! [$sc] final png has $cnt/$N -- ABORT"; exit 1; }
  echo "=== [$sc] 3-seed video ready ($cnt imgs) ==="
  PAIRS="$PAIRS $sc=$E/png"
done

ZIP=/mnt/d/avv/submissions/sub_round16_4seedtower_3seedvideo.zip
python build_submission_zip.py --scene_dirs $PAIRS --data_root $PRI --out $ZIP \
  || { echo "!!! BUILD_ZIP FAILED"; exit 1; }
python scripts/verify_zip.py --zip $ZIP --data_root $PRI || { echo "!!! VERIFY FAILED"; exit 1; }

cat > ${ZIP%.zip}.PROVENANCE.txt <<'EOP'
sub_round16_4seedtower_3seedvideo.zip -- round16 candidate (all-scene seed-extension vs r15):
  towers: UT 60k/8M 4-SEED ensemble (ut7+ut42+ut13+ut77, w=1/4 each) + DIS lens field.
  chair:  AA no-UT 60k lpearly 3-SEED ensemble (aa42+aa7+aa13, w=1/3), no field.
  bonsai: AA no-UT 30k capD+lpearly 3-SEED ensemble (aa42+aa7+aa13, w=1/3), no field.
  vs r15: towers 3->4 seed, chair/bonsai 2->3 seed. ALL 7 scenes changed this time (not a
  clean single-axis swap) -- if this needs isolating later, resubmit r15-towers-only vs
  r15-videos-only. Baseline to beat: r15 = 76.95340.
EOP
echo "=== ROUND16 BUILT -> $ZIP ==="
ls -la $ZIP

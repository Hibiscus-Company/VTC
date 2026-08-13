#!/bin/bash
# r20_combined AUTO-BUILD v3 -- ADD composition, minus HCM0674 (its EMA run went pathological
# ~19h and was killed 24/07 05:17; it stays at r16's 4-member ensemble). The other 4 towers get
# the 5-member add (orig ut7 + ut7_ema999 + 3 seeds); HCM0421 gets a 6th (ut7_ema099); chair
# gets the 6-member add (aa trio + ema trio); bonsai = r16.
# Waits ONLY on HCM0540 (the last EMA tower still training on GPU1). CPU-only. NEVER submits.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
PRI=/mnt/d/avv/data/phase1/private_set2
R9=/mnt/d/avv/r2r9

while [ ! -f /mnt/d/avv/tower_ema_rollout_HCM0540.DONE ]; do sleep 60; done
echo "=== HCM0540 marker present, building r20 (ADD composition, HCM0674 at r16) $(date) ==="
conda activate fastgs2

# chair 6-member
E=/mnt/d/avv/r20/video_ens/chair6
mkdir -p $E
python ensemble_renders.py \
  --dirs /mnt/d/avv/r14/chair_aa42/test_png /mnt/d/avv/r14/chair_aa7/test_png /mnt/d/avv/r14/chair_aa13/test_png \
         /mnt/d/avv/r17/chair_ema099_seed42/test_png /mnt/d/avv/r17/chair_ema099_seed7/test_png /mnt/d/avv/r17/chair_ema099_seed13/test_png \
  --weights 0.166667 0.166667 0.166667 0.166667 0.166667 0.166667 \
  --masks none none none none none none \
  --out $E/jpg --png_dir $E/png \
  --names_from $PRI/chair/test/test_poses.csv || { echo "!!! [chair6] ENSEMBLE FAILED"; exit 1; }
PAIRS="chair=$E/png bonsai=/mnt/d/avv/r16/video_ens/bonsai/png"

# HCM0674 stays at r16 (no EMA member)
PAIRS="$PAIRS HCM0674=/mnt/d/avv/r16/tower_ens/HCM0674/png"

# the 4 EMA-boosted towers
for S in HCM0421 HCM0539 HCM0540 HCM0644; do
  E=/mnt/d/avv/r20/tower_ens/$S
  mkdir -p $E
  n=$(ls /mnt/d/avv/r17/${S}_ut7_ema999/test_png 2>/dev/null | wc -l)
  [ "$n" -eq 60 ] || { echo "!!! [$S] ema999 member has $n/60 -- ABORT"; exit 1; }
  EXTRA=""; W="0.2 0.2 0.2 0.2 0.2"; M="none none none none none"
  if [ "$S" = "HCM0421" ]; then
    EXTRA="/mnt/d/avv/r17/HCM0421_ut7_ema099/test_png"
    W="0.166667 0.166667 0.166667 0.166667 0.166667 0.166667"
    M="none none none none none none"
  fi
  python ensemble_renders.py \
    --dirs $R9/models/${S}_ut7/test_png /mnt/d/avv/r17/${S}_ut7_ema999/test_png \
           $R9/models/${S}_ut42/test_png $R9/models/${S}_ut13/test_png $R9/models/${S}_ut77/test_png $EXTRA \
    --weights $W --masks $M \
    --out $E/jpg --png_dir $E/png_ens \
    --names_from $PRI/$S/test/test_poses.csv || { echo "!!! [$S] ENSEMBLE FAILED"; exit 1; }
  python gsplat_track/apply_field.py --strict \
    --in_dir $E/png_ens --field $R9/fields/$S.npy --out_dir $E/png \
    || { echo "!!! [$S] APPLY_FIELD FAILED"; exit 1; }
  cnt=$(ls $E/png/*.png 2>/dev/null | wc -l)
  [ "$cnt" -eq 60 ] || { echo "!!! [$S] final png $cnt/60 -- ABORT"; exit 1; }
  echo "=== [$S] ADD-composition tower ready ==="
  PAIRS="$PAIRS $S=$E/png"
done

ZIP=/mnt/d/avv/submissions/sub_round20_emaadd.zip
python build_submission_zip.py --scene_dirs $PAIRS --data_root $PRI --out $ZIP \
  || { echo "!!! BUILD_ZIP FAILED"; exit 1; }
python scripts/verify_zip.py --zip $ZIP --data_root $PRI || { echo "!!! VERIFY FAILED"; exit 1; }

cat > ${ZIP%.zip}.PROVENANCE.txt <<'EOP'
sub_round20_emaadd.zip -- round20 COMBINED, ADD composition (user: "stop fragment improvement";
eval evidence 23/07: ADD beats REPLACE by +0.6 scene-level, 6-member chair beats 4-member,
pixel-mean beats median/agreement):
  towers HCM0421/0539/0540/0644: 5-member = ut7 + ut7_ema999(decay .999) + ut42 + ut13 + ut77,
    w=1/5, same DIS field. HCM0421 has a 6th member (ut7_ema099, w=1/6).
  tower HCM0674: UNCHANGED from r16 (its EMA run went pathological ~19h, killed 24/07; will be
    added in a later round if re-run cheaply).
  chair: 6-member = aa42+aa7+aa13 (no EMA) + ema099 seed42/7/13, w=1/6, no field.
  bonsai: byte-identical to r16.
  Baseline: r16 = 77.09550. Fragments r17 +0.0024 / r18 +0.0028 (dilution-null, superseded).
  Expected: +0.1-0.2 blended (composition-class transfer >=1x; 4/5 towers + full chair upgraded).
EOP
echo "=== R20_EMAADD BUILT + VERIFIED $(date) -> $ZIP ==="
touch /mnt/d/avv/r20_combined.BUILT

#!/bin/bash
# r28 = r27 + whatever mip3d members landed from the r28 member bank.
#
# COMPOSITION IS THE ONE LEVER THAT HAS NEVER TRANSFERRED BELOW 1x IN 12 GRADED ROUNDS. Every
# post-processing axis is now closed (this session alone: resampling kernel -> shipped as r27,
# band-limited warp, field estimator, per-image exposure, member alignment, chair field, photo-reuse
# on the video scenes).
#
# ONLY scenes with a landed member are rebuilt; everything else is copied BYTE-IDENTICALLY out of
# r27, and the build asserts that the set of changed scenes is exactly the set it intended to change.
#
# WEIGHTS, and why they differ per scene:
#   towers   0.667 * (r22 6-member mean) + 0.167 * mip555 + 0.167 * mip777
#            = mip3d family at 0.333 total. Justified: the HCM0421 2-blend sweep put a mip3d member
#            at w=0.60 against ONE plain member, i.e. worth 1.5 plain members, so 2*1.5/(6+2*1.5).
#   chair    UNIFORM over 8 (7 existing + mip3d). bonsai UNIFORM over 7 (6 existing + mip3d).
#            The 1.5x factor was measured on a TOWER and we cannot verify it on the video scenes
#            (no test GT), and uniform is the proven optimum WITHIN a family -- so take the
#            conservative weight rather than extrapolate a bonus we have not measured here.
#
# Member lists are the byte-verified ones in /mnt/d/avv/ENSEMBLE_MANIFEST.md (chair reconstructs
# EXACTLY, bonsai to rounding, both re-encode 58/58 and 28/28 byte-identical to the shipped zip).
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
MEM=/mnt/d/avv/r28_members
OUT=/mnt/d/avv/r28
FLD=/mnt/d/avv/fields_median
mkdir -p $OUT
CHANGED=""

# ---------------------------------------------------------------- towers
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  if [ ! -f $MEM/${T}.DONE ]; then echo "skip $T (no new member)"; continue; fi
  n=$(ls $MEM/$T/test_png 2>/dev/null | wc -l)
  if [ "$n" -ne 60 ]; then echo "!!! skip $T: member has $n/60 renders"; continue; fi
  echo "=== $T: 6-member(0.667) + mip555(0.167) + mip777(0.167) ==="
  mkdir -p $OUT/tower_ens/$T
  python ensemble_renders.py \
    --dirs /mnt/d/avv/r22/tower_ens/$T/png_ens /mnt/d/avv/r25_mip3d/$T/test_png $MEM/$T/test_png \
    --weights 4 1 1 \
    --out $OUT/tower_ens/$T/jpg_tmp --png_dir $OUT/tower_ens/$T/png_ens \
    --names_from $DATA/$T/test/test_poses.csv
  python gsplat_track/apply_field.py --in_dir $OUT/tower_ens/$T/png_ens \
    --field $FLD/$T.npy --out_dir $OUT/tower_ens/$T/png --strict
  CHANGED="$CHANGED $T"
done

# ---------------------------------------------------------------- chair (8 uniform, NO field)
if [ -f $MEM/chair.DONE ] && [ "$(ls $MEM/chair/test_png 2>/dev/null | wc -l)" -eq 58 ]; then
  echo "=== chair: 7 existing + mip3d, uniform over 8, no field ==="
  mkdir -p $OUT/video_ens/chair
  python ensemble_renders.py \
    --dirs /mnt/d/avv/r14/chair_aa42/test_png /mnt/d/avv/r14/chair_aa7/test_png \
           /mnt/d/avv/r14/chair_aa13/test_png /mnt/d/avv/r17/chair_ema099_seed42/test_png \
           /mnt/d/avv/r17/chair_ema099_seed7/test_png /mnt/d/avv/r17/chair_ema099_seed13/test_png \
           /mnt/d/avv/r17/chair_depth_seed42/test_png $MEM/chair/test_png \
    --out $OUT/video_ens/chair/jpg_tmp --png_dir $OUT/video_ens/chair/png \
    --names_from $DATA/chair/test/test_poses.csv
  CHANGED="$CHANGED chair"
else
  echo "skip chair (no new member)"
fi

# ---------------------------------------------------------------- bonsai (7 uniform, NO field)
if [ -f $MEM/bonsai.DONE ] && [ "$(ls $MEM/bonsai/test_png 2>/dev/null | wc -l)" -eq 28 ]; then
  echo "=== bonsai: 6 existing + mip3d, uniform over 7, no field ==="
  mkdir -p $OUT/video_ens/bonsai
  python ensemble_renders.py \
    --dirs /mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
           /mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png \
           /mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png \
           $MEM/bonsai/test_png \
    --out $OUT/video_ens/bonsai/jpg_tmp --png_dir $OUT/video_ens/bonsai/png \
    --names_from $DATA/bonsai/test/test_poses.csv
  CHANGED="$CHANGED bonsai"
else
  echo "skip bonsai (no new member)"
fi

echo ""
echo "=== scenes to rebuild:$CHANGED ==="
if [ -z "$CHANGED" ]; then echo "NOTHING LANDED -- no r28"; exit 1; fi

echo "=== assemble on top of r27 ==="
CHANGED="$CHANGED" python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
R27 = "/mnt/d/avv/submissions/sub_round27_lanczos.zip"
OUT = "/mnt/d/avv/submissions/sub_round28_mip3dmembers.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q = {"HCM0421": 99, "HCM0539": 100, "HCM0540": 100, "HCM0644": 100, "HCM0674": 100,
     "chair": 100, "bonsai": 100}
SRC = {**{t: f"/mnt/d/avv/r28/tower_ens/{t}/png" for t in
          ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")},
       "chair": "/mnt/d/avv/r28/video_ens/chair/png",
       "bonsai": "/mnt/d/avv/r28/video_ens/bonsai/png"}
changed = set(os.environ["CHANGED"].split())
src = zipfile.ZipFile(R27)
n_enc = n_copy = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        if scene in changed:
            stem = os.path.splitext(os.path.basename(i.filename))[0]
            im = Image.open(os.path.join(SRC[scene], stem + ".png")).convert("RGB")
            kw = dict(SHIPPED); kw["quality"] = Q[scene]
            b = io.BytesIO(); im.save(b, "JPEG", **kw)
            z.writestr(i.filename, b.getvalue()); n_enc += 1
        else:
            z.writestr(i.filename, src.read(i.filename)); n_copy += 1
print(f"re-encoded {n_enc} files in {sorted(changed)}, copied {n_copy} verbatim")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round28_mip3dmembers.zip --data_root $DATA

echo "=== confirm ONLY the intended scenes differ from r27 ==="
CHANGED="$CHANGED" python3 - <<'PYEOF'
import zipfile, os
a = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round27_lanczos.zip")
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round28_mip3dmembers.zip")
assert set(a.namelist()) == set(b.namelist())
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
want = sorted(os.environ["CHANGED"].split())
print("scenes changed vs r27:", diff)
assert diff == want, f"MISMATCH: changed {diff} but intended {want}"
sz = os.path.getsize("/mnt/d/avv/submissions/sub_round28_mip3dmembers.zip") / 1e6
print(f"CONFIRMED: exactly the intended scenes changed. {sz:.1f} MB")
assert sz < 350, f"OVER BUDGET: {sz:.1f} MB"
PYEOF
touch /mnt/d/avv/build_r28.DONE
echo "=== R28 BUILD DONE $(date) ==="

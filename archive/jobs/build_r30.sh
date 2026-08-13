#!/bin/bash
# r30 = r29 + THREE measured changes. Base r29 = 77.6644.
# Usage: build_r30.sh
#
# ---------------------------------------------------------------------------------------------
# CHANGE 1: TOWERS GO 8 -> 10 MEMBERS, AT TRUE UNIFORM WEIGHT.   (+0.0488/tower -> +0.0349 blended)
# ---------------------------------------------------------------------------------------------
# The ensemble has an INTERIOR OPTIMUM at k~10. Measured on the production harness (HCM0181, real
# test GT, n=60, full shipped chain, members added best-first by solo PSNR):
#     k= 4  77.9190   k= 6  78.5059   k= 8  78.8165 (r29)   k=10  78.8653  <- argmax
#     k=12  78.8075   k=14  78.7380
#   8 -> 10 = +0.0488 +/- 0.0104 (4.7 sigma).  12 = -0.0091.  14 = -0.0785.
# MECHANISM, visible in the submetrics: PSNR rises MONOTONICALLY to k=14 (26.1849 -> 26.2448) while
# LPIPS TURNS OVER after k=10 (0.09338 -> 0.09659). Deeper pixel-averaging keeps cutting squared
# error while destroying perceptual texture, and LPIPS carries the -40 coefficient. This is the same
# "our HF is incoherent with GT" physics that makes JPEG beat lossless PNG and lanczos4 beat sharper
# kernels. => TAKE EXACTLY TWO MORE MEMBERS. Taking all six available would LOSE score.
#
# WHICH TWO -- and why NOT the ones that look obvious. All 30 unused members passed member_gate.py,
# but that gate only catches COLLAPSE, not mediocrity. Ranking them against the shipped 8-member
# ensemble (GT-free: consensus deviation + Laplacian-L0 HF energy, cand_rank.py):
#     shipped r2r9_ut7/ut42/seed101  dev 3.23-3.28  HF 1.037-1.039
#     shipped mip555                 dev 2.49       HF 1.014
#     r17_ema999    dev 2.79  HF 1.038   <- BETTER than every shipped UT member. PICK.
#     r2r8_ut42     dev 3.89  HF 1.013   <- credible second. PICK.
#     r2r8_ut7      dev 3.94  HF 1.015
#     s2g_champA/memB/memC  dev 7.35-7.41  HF 0.985-1.005  <- 2.3x the deviation of ANYTHING we
#       ship, and systematically SOFTER than the peer detail band on 14/15 tower-candidate pairs.
#       These are the members an agent recommended first ("champA"). Adding them would repeat the
#       gsplatB6bilagrid failure (-0.7418). EXCLUDED.
# WEIGHTS: `--weights 6 1 1 1 1` is TRUE 10-way uniform (the r22 dir is a 6-member mean). This also
# retires r29's `4 1 1`, which over-weighted each mip3d member 1.5x. Family tilt was measured at
# -0.0435 vs uniform at k=8, i.e. uniform is a genuine interior optimum, not a flat-curve midpoint.
#
# ---------------------------------------------------------------------------------------------
# CHANGE 2: GAUSSIAN-SMOOTH THE LENS FIELD, sigma=1 on the ds8 grid.  (+0.0106/tower -> +0.0076)
# ---------------------------------------------------------------------------------------------
# Measured on the FULL r29 chain, HCM0181, real test GT, n=60:
#     plain x1.30  78.5186 | gauss1 x1.30  78.5293  +0.0106 +/- 0.0007, 59/60 WINS
#                          | gauss2 x1.30  78.5307  +0.0120 +/- 0.0015, 52/60
# sigma=2 is nominally higher but the gap (+0.0014 +/- 0.0017) is noise and its win rate is worse.
# sigma=1 is the pick. Independently, the chair LOVO preferred med_ds8_g1_rlan (+0.0300) over
# med_ds8_rlan (+0.0207) -- same conclusion on a second scene. Fields in fields_median_g1_g130/.
#
# ---------------------------------------------------------------------------------------------
# CHANGE 3: CHAIR FINALLY GETS A LENS FIELD.                       (+0.0483 chair -> +0.0069)
# ---------------------------------------------------------------------------------------------
# chair and bonsai have shipped with NO field for the entire competition. chair's was killed by a
# leave-one-view-out test over TRAIN views. That protocol is now CALIBRATED against real test GT on
# HCM0181, same member, same field, same metric, only the pose set differing:
#     LOVO (held-out TRAIN views)  none 78.8906 -> field 79.8550 = +0.9644
#     PRODUCTION (real TEST GT)    none 75.2555 -> field 76.6670 = +1.4116   ratio 1.46x
# So LOVO UNDER-READS a field by 1.46x at matched gain, and gain 1.0 -> 1.30 adds a further 10.4%
# (76.6670 -> 76.8141). chair's LOVO +0.0299 therefore projects to ~+0.0483 in production.
# NOTE this REFUTES the larger hypothesis that LOVO under-reads by 10-30x: it reads field PRESENCE
# almost exactly right (+0.9644 vs the LB's +1.028/scene at R7). It gets AMPLITUDE wrong, not
# presence. chair genuinely has ~32x less misregistration than a tower -- mean |d| 0.101 px vs
# 0.211 px -- because it is a handheld camera at 720x1280, not a drone with k1~+0.009.
# Rule 10 clean: fit on chair TRAIN photos only, --strict passes, gt_dir is a train dir.
#
# ---------------------------------------------------------------------------------------------
# DELIBERATELY NOT DONE
# ---------------------------------------------------------------------------------------------
# * NO member change on chair/bonsai. The k=10 optimum was measured WITH energy restore lam=1.0,
#   which puts texture back. The video scenes ship lam=0, so their over-smoothing penalty per added
#   member is LARGER and their optimal k is probably LOWER, not higher. r28 already cost us a round
#   by assuming tower behaviour transfers to the video scenes. Towers only.
# * NO encode change. The byte cap is not a lever: 4:4:4 = -0.0076, 4:2:2 = -0.0035, chroma blur
#   monotonically negative, q96 = -0.1106. q98/q99/q100 are one flat plateau (+0.0020 +/- 0.0020),
#   so HCM0421's stray Q99 is harmless and "fixing" it would spend 6.64 MiB for nothing.
# * NO field amplitude past 1.30. Measured twice: 1.45 == 1.30 inside noise on both chains.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
BASE=/mnt/d/avv/submissions/sub_round29_members.zip
DATA=/mnt/d/avv/data/phase1/private_set2
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
OUT=/mnt/d/avv/r30
FLD=/mnt/d/avv/fields_median_g1_g130
CHAIRFLD=/mnt/d/avv/fields_chair/chair_g1_g130.npy
mkdir -p $OUT
CHANGED=""

[ -f "$BASE" ] || { echo "!!! base zip missing: $BASE"; exit 1; }

# ---------------------------------------------------------------- towers: 10 members, uniform
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  NEW1=/mnt/d/avv/r17/${T}_ut7_ema999/test_png
  NEW2=/mnt/d/avv/r2r8/models/${T}_ut42/test_png
  for d in "$NEW1" "$NEW2"; do
    [ "$(ls $d 2>/dev/null | wc -l)" -eq 60 ] || { echo "!!! $T: bad render count in $d -- SKIPPING SCENE"; continue 2; }
  done
  # resume: apply_field writes 60 PNGs PLUS field_applied.json, so count *.png only
  if [ "$(ls $OUT/tower_ens/$T/png/*.png 2>/dev/null | wc -l)" -eq 60 ]; then
    echo "=== $T already complete -- reusing ==="; CHANGED="$CHANGED $T"; continue
  fi
  echo "=== $T: 10 members uniform (6+1+1+1+1) -> restore(lam=1.0,k=10) -> field gauss1 x1.30 ==="
  mkdir -p $OUT/tower_ens/$T
  python ensemble_renders.py \
    --dirs /mnt/d/avv/r22/tower_ens/$T/png_ens /mnt/d/avv/r25_mip3d/$T/test_png \
           /mnt/d/avv/r28_members/$T/test_png $NEW1 $NEW2 \
    --weights 6 1 1 1 1 \
    --out $OUT/tower_ens/$T/jpg_tmp --png_dir $OUT/tower_ens/$T/png_ens \
    --names_from $DATA/$T/test/test_poses.csv
  python $ER --mode apply --k 10 --lam 1.0 \
    --ens_dir $OUT/tower_ens/$T/png_ens \
    --member_dirs /mnt/d/avv/r2r9/models/${T}_ut7/test_png /mnt/d/avv/r2r9/models/${T}_ut13/test_png \
                  /mnt/d/avv/r2r9/models/${T}_ut42/test_png /mnt/d/avv/r2r9/models/${T}_ut77/test_png \
                  /mnt/d/avv/r22_seed101/${T}/test_png /mnt/d/avv/r25_mip3d/${T}/test_png \
                  /mnt/d/avv/r28_members/$T/test_png $NEW1 $NEW2 \
    --out_dir $OUT/tower_ens/$T/png_er
  python gsplat_track/apply_field.py --in_dir $OUT/tower_ens/$T/png_er \
    --field $FLD/$T.npy --out_dir $OUT/tower_ens/$T/png --strict
  [ "$(ls $OUT/tower_ens/$T/png/*.png 2>/dev/null | wc -l)" -eq 60 ] || { echo "!!! $T: field output count wrong"; exit 1; }
  CHANGED="$CHANGED $T"
done

# ---------------------------------------------------------------- chair: r29 pixels + the field
echo "=== chair: r29 8-member mean (lam=0) -> lens field gauss1 x1.30 ==="
mkdir -p $OUT/video_ens/chair
python gsplat_track/apply_field.py --in_dir /mnt/d/avv/r29/video_ens/chair/png \
  --field $CHAIRFLD --out_dir $OUT/video_ens/chair/png --strict
[ "$(ls $OUT/video_ens/chair/png/*.png 2>/dev/null | wc -l)" -eq 58 ] || { echo "!!! chair: field output count wrong"; exit 1; }
CHANGED="$CHANGED chair"

# bonsai: UNCHANGED, inherited verbatim from r29.

echo ""
echo "=== scenes rebuilt:$CHANGED ==="
[ -n "$CHANGED" ] || { echo "NOTHING LANDED -- no r30"; exit 1; }

echo "=== assemble on top of $(basename $BASE) ==="
CHANGED="$CHANGED" BASE="$BASE" python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
BASE = os.environ["BASE"]
OUT = "/mnt/d/avv/submissions/sub_round30_k10.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q = {"HCM0421": 99, "HCM0539": 100, "HCM0540": 100, "HCM0644": 100, "HCM0674": 100,
     "chair": 100, "bonsai": 100}
SRC = {**{t: f"/mnt/d/avv/r30/tower_ens/{t}/png" for t in
          ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")},
       "chair": "/mnt/d/avv/r30/video_ens/chair/png"}
changed = set(os.environ["CHANGED"].split())
missing = [s for s in changed if s not in SRC]
assert not missing, f"changed scene with no source dir: {missing}"
src = zipfile.ZipFile(BASE)
n = c = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        if scene in changed:
            stem = os.path.splitext(os.path.basename(i.filename))[0]
            p = os.path.join(SRC[scene], stem + ".png")
            assert os.path.exists(p), f"missing rebuilt PNG: {p}"
            im = Image.open(p).convert("RGB")
            kw = dict(SHIPPED); kw["quality"] = Q[scene]
            b = io.BytesIO(); im.save(b, "JPEG", **kw)
            z.writestr(i.filename, b.getvalue()); n += 1
        else:
            z.writestr(i.filename, src.read(i.filename)); c += 1
print(f"re-encoded {n} files in {sorted(changed)}, copied {c} verbatim")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round30_k10.zip --data_root $DATA

echo "=== change + size check ==="
CHANGED="$CHANGED" BASE="$BASE" python3 - <<'PYEOF'
import zipfile, os
a = zipfile.ZipFile(os.environ["BASE"])
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round30_k10.zip")
assert set(a.namelist()) == set(b.namelist()), "name set changed"
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
want = sorted(set(os.environ["CHANGED"].split()))
print("scenes changed vs r29:", diff)
assert diff == want, f"MISMATCH: changed {diff}, intended {want}"
nb = os.path.getsize("/mnt/d/avv/submissions/sub_round30_k10.zip")
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB = {nb/1e6:.2f} MB")
print(f"  vs MiB cap 367,001,600 -> {'FITS' if nb < 367_001_600 else 'OVER'}")
assert nb < 367_001_600, f"OVER the MiB cap: {nb:,}"
PYEOF
touch /mnt/d/avv/build_r30.DONE
echo "=== r30 BUILT -- NOT SUBMITTED ==="

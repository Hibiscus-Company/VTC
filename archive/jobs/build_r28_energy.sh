#!/bin/bash
# r28 = r27 + ENSEMBLE ENERGY RESTORATION, inserted between the ensemble and the lens field.
# Usage: build_r28_energy.sh <tower_lam> <chair_lam> <bonsai_lam>   (a bonsai_lam of 0 ships r27's
# bonsai byte-identically)
#
# LAMBDA IS CHOSEN BY SCORE. It is still allocated PER SCENE because the towers
# are 298.64 of the 345.04 MB (86.6%) while carrying the SMALLER per-scene gain; chair+bonsai are
# only 46.40 MB and carry the larger one (eval-split: chair +0.536, bonsai +0.221 at lam=1.0, vs
# tower +0.226). Byte growth per unit lambda is scene-dependent (at lam=0.5: towers 0.51%, chair
# 1.90%, bonsai 7.59%), so lambda is per-scene -- but the BUDGET NO LONGER BINDS. The organiser's
# 350MB is 350 MiB = 367,001,600 bytes, not 350e6; r27 sits at 329.11 MiB, leaving 20.89 MiB.
# Even lambda=2.0 on all seven scenes fits (346.35 MiB). So lambda is chosen by SCORE, not bytes.
#
# THE DEFECT: averaging k members destroys local high-frequency energy wherever they disagree, and
# the deficit is not uniform -- it is the ensemble's own disagreement map. Measured on the harness:
# the pixel mean sits 15% below GT energy at the finest Laplacian level, and ALL of the headroom is
# at that level (levels >=1 are already at GT energy).
# THE FIX: rescale only the finest band by r = sqrt(1 + (k/(k-1)) * E(L0_i - L0_mean) / E(L0_mean)).
#
# WHY THIS IS NOT THE SHARPENING LEVER THAT ALREADY DIED: the boost is spatially varying and driven
# by the members, landing where averaging destroyed energy (corr(r, log E_mean) = -0.63), not on
# edges. The MSE-optimal global unsharp was -0.618 and the best radial linear filter is bounded at
# +0.043; this is +0.15 because it is not a linear filter.
# CONTROLS (all on the production harness, real test GT):
#   real disagreement map          +0.1988
#   same histogram, shuffled       -0.9766      <- the map is load-bearing
#   same histogram, ordered +E     -2.4921      <- the SIGN of the map is load-bearing
#   same mean boost, no map        -0.0870      <- it is not a global gain
# ANTI-CLEANUP SIGNATURE: the gain GROWS with ensemble depth (k=2 +0.099 ... k=6 +0.244). EMA and
# the JPEG "fix" both DECAYED with depth and then inverted in production. Production ships 7
# members, so the k=4 harness number is a lower bound.
# CROSS-SCENE: positive on tower/chair/bonsai at every lambda tested; the eval-split-to-production
# regime factor measured 0.88, so it is not proxy-inflated.
# Two independent agents measured this lever separately and agreed (+0.1505 and +0.151).
# OUT is a FRESH directory. The first attempt at this build ran at lambda=0.75 with a 4-member map
# and got through three towers before it was killed; those stale PNGs were still on disk. A build
# that writes into a dirty directory can silently assemble a zip mixing two lambdas -- an error that
# is invisible in the output and unattributable on the leaderboard. Fresh dir, no deletion needed.
set -e
TLAM=${1:?usage: build_r28_energy.sh <tower_lam> <chair_lam> <bonsai_lam>}
CLAM=${2:?usage: build_r28_energy.sh <tower_lam> <chair_lam> <bonsai_lam>}
BLAM=${3:?usage: build_r28_energy.sh <tower_lam> <chair_lam> <bonsai_lam>}
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
OUT=/mnt/d/avv/r28e_v2
FLD=/mnt/d/avv/fields_median
mkdir -p $OUT

for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  echo "=== $T: energy restore (lam=$TLAM, k=7, 6-member map) -> median field (lanczos) ==="
  mkdir -p $OUT/tower_ens/$T
  python $ER --mode apply --k 7 --lam $TLAM \
    --ens_dir /mnt/d/avv/r25/tower_ens/$T/png_ens \
    --member_dirs /mnt/d/avv/r2r9/models/${T}_ut7/test_png \
                  /mnt/d/avv/r2r9/models/${T}_ut13/test_png \
                  /mnt/d/avv/r2r9/models/${T}_ut42/test_png \
                  /mnt/d/avv/r2r9/models/${T}_ut77/test_png \
                  /mnt/d/avv/r22_seed101/${T}/test_png \
                  /mnt/d/avv/r25_mip3d/${T}/test_png \
    --out_dir $OUT/tower_ens/$T/png_er
  python gsplat_track/apply_field.py --in_dir $OUT/tower_ens/$T/png_er \
    --field $FLD/$T.npy --out_dir $OUT/tower_ens/$T/png --strict
done

echo "=== chair: energy restore (lam=$CLAM, k=7), NO field ==="
mkdir -p $OUT/video_ens
python $ER --mode apply --k 7 --lam $CLAM \
  --ens_dir /mnt/d/avv/r21/video_ens/chair7/png \
  --member_dirs /mnt/d/avv/r14/chair_aa42/test_png /mnt/d/avv/r14/chair_aa7/test_png \
                /mnt/d/avv/r14/chair_aa13/test_png /mnt/d/avv/r17/chair_ema099_seed42/test_png \
                /mnt/d/avv/r17/chair_ema099_seed7/test_png /mnt/d/avv/r17/chair_ema099_seed13/test_png \
                /mnt/d/avv/r17/chair_depth_seed42/test_png \
  --out_dir $OUT/video_ens/chair/png

if [ "$BLAM" != "0" ]; then
  echo "=== bonsai: energy restore (lam=$BLAM, k=6), NO field ==="
  python $ER --mode apply --k 6 --lam $BLAM \
    --ens_dir /mnt/d/avv/r24/bonsai/png \
    --member_dirs /mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
                  /mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png \
                  /mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png \
    --out_dir $OUT/video_ens/bonsai/png
else
  echo "=== bonsai: lam=0, shipping r27 bytes verbatim (worst score-per-MB of the seven) ==="
fi

echo "=== assemble on top of r27 (ALL 7 scenes change) ==="
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
R27 = "/mnt/d/avv/submissions/sub_round27_lanczos.zip"
OUT = "/mnt/d/avv/submissions/sub_round28_energy.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q = {"HCM0421": 99, "HCM0539": 100, "HCM0540": 100, "HCM0644": 100, "HCM0674": 100,
     "chair": 100, "bonsai": 100}
SRC = {**{t: f"/mnt/d/avv/r28e_v2/tower_ens/{t}/png" for t in
          ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")},
       "chair": "/mnt/d/avv/r28e_v2/video_ens/chair/png",
       "bonsai": "/mnt/d/avv/r28e_v2/video_ens/bonsai/png"}
SKIP = set(os.environ.get("SKIP_SCENES", "").split())
src = zipfile.ZipFile(R27)
n = c = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        if scene in SKIP:
            z.writestr(i.filename, src.read(i.filename)); c += 1
            continue
        stem = os.path.splitext(os.path.basename(i.filename))[0]
        im = Image.open(os.path.join(SRC[scene], stem + ".png")).convert("RGB")
        kw = dict(SHIPPED); kw["quality"] = Q[scene]
        b = io.BytesIO(); im.save(b, "JPEG", **kw)
        z.writestr(i.filename, b.getvalue()); n += 1
print(f"re-encoded {n} files, copied {c} verbatim (skipped: {sorted(SKIP)})")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round28_energy.zip --data_root $DATA

echo "=== size + change check ==="
SKIP_SCENES="$SKIP" python3 - <<'PYEOF'
import zipfile, os, collections
a = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round27_lanczos.zip")
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round28_energy.zip")
assert set(a.namelist()) == set(b.namelist())
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
want = sorted(set("HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 chair bonsai".split())
              - set(os.environ.get("SKIP_SCENES", "").split()))
print("scenes changed vs r27:", diff)
assert diff == want, f"MISMATCH: changed {diff}, intended {want}"
per = collections.Counter()
for n in b.namelist():
    per[n.split("/")[0]] += b.getinfo(n).file_size
for s in sorted(per):
    print(f"   {s:>9} {per[s]/1e6:7.2f} MB   (r27 {sum(a.getinfo(n).file_size for n in a.namelist() if n.startswith(s+'/'))/1e6:7.2f})")
nb = os.path.getsize("/mnt/d/avv/submissions/sub_round28_energy.zip")
CAP = 350 * 1024 * 1024          # the organiser's 350MB is MiB: 367,001,600 bytes, not 350e6
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB / {nb/1e6:.2f} MB   "
      f"(cap {CAP:,} B = 350.00 MiB; r27 was {345092353/1048576:.2f} MiB)")
print(f"HEADROOM LEFT {(CAP-nb)/1048576:.2f} MiB")
assert nb < CAP, f"OVER BUDGET: {nb/1048576:.2f} MiB -- rebuild with a smaller lambda"
PYEOF
touch /mnt/d/avv/build_r28e.DONE
echo "=== R28 ENERGY BUILD DONE (tower=$TLAM chair=$CLAM bonsai=$BLAM) $(date) ==="

#!/bin/bash
# r29 = r28 + the mip3d members that landed, with the energy-restoration operator RE-DERIVED.
# Usage: build_r29.sh <base_zip> <tower_lam> <chair_lam> <bonsai_lam>
#   e.g. build_r29.sh /mnt/d/avv/submissions/sub_round28_energy.zip 1.0 1.25 1.0
#        build_r29.sh /mnt/d/avv/submissions/sub_round28_energy_decimalsafe.zip 0.5 0.5 0
#
# WHY THE OPERATOR MUST BE RE-DERIVED, NOT REUSED: adding a member changes BOTH inputs to energy
# restoration -- the ensemble mean it acts on, and the disagreement map it is driven by (a new
# member contributes its own deviations, and k rises so the k/(k-1) correction shrinks). Splicing
# a new member into r28's already-restored pixels would double-count. Every touched scene is
# rebuilt from members -> mean -> restore -> field -> JPEG.
#
# COMPOSITION IS THE LEVER: an agent measured members 4->6 on the production harness at +0.2608,
# consistent with the calibrated +0.0919 3->4 marginal, and called it "the highest-yield axis
# available". chair and bonsai have never had a mip3d member at all, and on those two scenes mip3d
# sits on top of gsplat's ANTIALIASED rasteriser (they train without --ut), which is the 2D
# screen-space filter -- so 3D + 2D together is full Mip-Splatting, never run before on this set.
#
# LAMBDA=0 ON A VIDEO SCENE STILL REBUILDS IT -- it gets the new member but no energy restoration.
# r28 graded 77.5029 (+0.1005) and its submetric split showed LPIPS transferred at 95% while SSIM
# went the WRONG WAY (-0.105pp vs +0.11pp predicted). Backing out the harness tower prediction
# (+0.134 blended) against the +0.1005 delivered puts chair+bonsai at roughly -0.02..-0.03 blended,
# i.e. NET NEGATIVE. Both were shipped one notch above their probable production optimum on the
# reasoning that the curve is flat near the top -- true on towers, false on the video scenes, and
# bonsai's disagreement map was already saturating. So the video scenes ship lam=0 in r29.
#
# FIELD AMPLITUDE GAIN 1.30 (the campaign's biggest find). A TRAIN-fitted field systematically
# UNDERSHOOTS: at train poses the model has already absorbed part of the misregistration into its
# own geometry, so DIS flow sees only the unabsorbed remainder; at NOVEL poses the absorbed warp
# does not cancel and the full displacement appears. Invisible to every train-side diagnostic we
# own -- our LOVO protocol says the optimum is 1.00 and is CONTRADICTED on the public scenes where
# real test GT exists.
# Two independent routes agree: scored on the production harness through the full chain, gain 1.30
# = +0.1328 (41/45 wins, all three submetrics up together); and a scoreless geometric decomposition
# giving best scalar alpha = 1.3185 (R2 0.901 -> 0.957), extended by the verifier to alpha* =
# 1.32-1.43 on all five public towers. Independent replication with the repo's own production code:
# +0.1036 +/- 0.0128, t=8.09. Verifier cut the blended magnitude to ~+0.05..+0.07 (HCM0181 is the
# best tower; production chain factor ~0.89; private fields are 18% weaker than public).
# The plateau is 1.25-1.40; 1.60 costs -0.05 and 1.80 costs -0.13, so do NOT overshoot. 1.30 is the
# conservative argmax. Rule 10 clean: fields are still fit on private TRAIN photos only and
# apply_field --strict still passes; only a SCALAR is calibrated on public test GT, the same
# sanctioned surface that set lanczos4 and lam=1.0.
#
# EVERY NEW MEMBER MUST PASS THE COLLAPSE GATE FIRST. bonsai has previously trained to completion,
# written 28 valid PNGs, and been FOG (median opacity 0.000). mip3d rescales every gaussian and
# compensates opacity, so that failure mode is live again on exactly the scene that had it.
set -e
BASE=${1:?usage: build_r29.sh <base_zip> <tower_lam> <chair_lam> <bonsai_lam>}
TLAM=${2:?}; CLAM=${3:?}; BLAM=${4:?}
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
MEM=/mnt/d/avv/r28_members
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
GATE=/home/bkai/.claude/jobs/1c9cf7e9/tmp/lens/member_gate.py
OUT=/mnt/d/avv/r29
FLD=/mnt/d/avv/fields_median_g130   # fields scaled by the measured 1.30 amplitude gain
mkdir -p $OUT
CHANGED=""

CHAIR_M="/mnt/d/avv/r14/chair_aa42/test_png /mnt/d/avv/r14/chair_aa7/test_png \
/mnt/d/avv/r14/chair_aa13/test_png /mnt/d/avv/r17/chair_ema099_seed42/test_png \
/mnt/d/avv/r17/chair_ema099_seed7/test_png /mnt/d/avv/r17/chair_ema099_seed13/test_png \
/mnt/d/avv/r17/chair_depth_seed42/test_png"
BONSAI_M="/mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
/mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png \
/mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png"

# ---------------------------------------------------------------- towers
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  [ -f $MEM/${T}.DONE ] || { echo "skip $T (no new member)"; continue; }
  [ "$(ls $MEM/$T/test_png 2>/dev/null | wc -l)" -eq 60 ] || { echo "!!! skip $T: bad render count"; continue; }
  echo "=== $T gate ==="
  python $GATE --candidate $MEM/$T/test_png --expect 60 \
    --peers /mnt/d/avv/r25_mip3d/$T/test_png /mnt/d/avv/r2r9/models/${T}_ut42/test_png \
            /mnt/d/avv/r22_seed101/$T/test_png \
    || { echo "!!! $T FAILED THE COLLAPSE GATE -- excluded"; continue; }
  echo "=== $T: 6-member(0.667) + mip555(0.167) + mip777(0.167) -> restore(lam=$TLAM,k=8) -> field ==="
  mkdir -p $OUT/tower_ens/$T
  python ensemble_renders.py \
    --dirs /mnt/d/avv/r22/tower_ens/$T/png_ens /mnt/d/avv/r25_mip3d/$T/test_png $MEM/$T/test_png \
    --weights 4 1 1 \
    --out $OUT/tower_ens/$T/jpg_tmp --png_dir $OUT/tower_ens/$T/png_ens \
    --names_from $DATA/$T/test/test_poses.csv
  python $ER --mode apply --k 8 --lam $TLAM \
    --ens_dir $OUT/tower_ens/$T/png_ens \
    --member_dirs /mnt/d/avv/r2r9/models/${T}_ut7/test_png /mnt/d/avv/r2r9/models/${T}_ut13/test_png \
                  /mnt/d/avv/r2r9/models/${T}_ut42/test_png /mnt/d/avv/r2r9/models/${T}_ut77/test_png \
                  /mnt/d/avv/r22_seed101/${T}/test_png /mnt/d/avv/r25_mip3d/${T}/test_png \
                  $MEM/$T/test_png \
    --out_dir $OUT/tower_ens/$T/png_er
  python gsplat_track/apply_field.py --in_dir $OUT/tower_ens/$T/png_er \
    --field $FLD/$T.npy --out_dir $OUT/tower_ens/$T/png --strict
  CHANGED="$CHANGED $T"
done

# ---------------------------------------------------------------- chair (8 uniform, NO field)
if [ -f $MEM/chair.DONE ] && [ "$(ls $MEM/chair/test_png 2>/dev/null | wc -l)" -eq 58 ]; then
  echo "=== chair gate ==="
  if python $GATE --candidate $MEM/chair/test_png --expect 58 --peers $CHAIR_M; then
    echo "=== chair: 8 uniform -> restore(lam=$CLAM,k=8), no field ==="
    mkdir -p $OUT/video_ens/chair
    python ensemble_renders.py --dirs $CHAIR_M $MEM/chair/test_png \
      --out $OUT/video_ens/chair/jpg_tmp --png_dir $OUT/video_ens/chair/png_ens \
      --names_from $DATA/chair/test/test_poses.csv
    if [ "$CLAM" != "0" ]; then
      python $ER --mode apply --k 8 --lam $CLAM --ens_dir $OUT/video_ens/chair/png_ens \
        --member_dirs $CHAIR_M $MEM/chair/test_png --out_dir $OUT/video_ens/chair/png
    else
      echo "    chair lam=0: shipping the plain 8-member mean, NO energy restoration"
      rm -rf $OUT/video_ens/chair/png; cp -r $OUT/video_ens/chair/png_ens $OUT/video_ens/chair/png
    fi
    CHANGED="$CHANGED chair"
  else
    echo "!!! chair FAILED THE COLLAPSE GATE -- excluded"
  fi
else
  echo "skip chair (no new member)"
fi

# ---------------------------------------------------------------- bonsai (7 uniform, NO field)
if [ -f $MEM/bonsai.DONE ] && [ "$(ls $MEM/bonsai/test_png 2>/dev/null | wc -l)" -eq 28 ]; then
  echo "=== bonsai gate ==="
  if python $GATE --candidate $MEM/bonsai/test_png --expect 28 --peers $BONSAI_M; then
    echo "=== bonsai: 7 uniform -> restore(lam=$BLAM,k=7), no field ==="
    mkdir -p $OUT/video_ens/bonsai
    python ensemble_renders.py --dirs $BONSAI_M $MEM/bonsai/test_png \
      --out $OUT/video_ens/bonsai/jpg_tmp --png_dir $OUT/video_ens/bonsai/png_ens \
      --names_from $DATA/bonsai/test/test_poses.csv
    if [ "$BLAM" != "0" ]; then
      python $ER --mode apply --k 7 --lam $BLAM --ens_dir $OUT/video_ens/bonsai/png_ens \
        --member_dirs $BONSAI_M $MEM/bonsai/test_png --out_dir $OUT/video_ens/bonsai/png
    else
      echo "    bonsai lam=0: shipping the plain 7-member mean, NO energy restoration"
      rm -rf $OUT/video_ens/bonsai/png; cp -r $OUT/video_ens/bonsai/png_ens $OUT/video_ens/bonsai/png
    fi
    CHANGED="$CHANGED bonsai"
  else
    echo "!!! bonsai FAILED THE COLLAPSE GATE -- excluded"
  fi
else
  echo "skip bonsai (no new member)"
fi

echo ""
echo "=== scenes rebuilt:$CHANGED ==="
[ -n "$CHANGED" ] || { echo "NOTHING LANDED -- no r29"; exit 1; }

echo "=== assemble on top of $(basename $BASE) ==="
CHANGED="$CHANGED" BASE="$BASE" python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
BASE = os.environ["BASE"]
OUT = "/mnt/d/avv/submissions/sub_round29_members.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q = {"HCM0421": 99, "HCM0539": 100, "HCM0540": 100, "HCM0644": 100, "HCM0674": 100,
     "chair": 100, "bonsai": 100}
SRC = {**{t: f"/mnt/d/avv/r29/tower_ens/{t}/png" for t in
          ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")},
       "chair": "/mnt/d/avv/r29/video_ens/chair/png",
       "bonsai": "/mnt/d/avv/r29/video_ens/bonsai/png"}
changed = set(os.environ["CHANGED"].split())
src = zipfile.ZipFile(BASE)
n = c = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        if scene in changed:
            stem = os.path.splitext(os.path.basename(i.filename))[0]
            im = Image.open(os.path.join(SRC[scene], stem + ".png")).convert("RGB")
            kw = dict(SHIPPED); kw["quality"] = Q[scene]
            b = io.BytesIO(); im.save(b, "JPEG", **kw)
            z.writestr(i.filename, b.getvalue()); n += 1
        else:
            z.writestr(i.filename, src.read(i.filename)); c += 1
print(f"re-encoded {n} files in {sorted(changed)}, copied {c} verbatim")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round29_members.zip --data_root $DATA

echo "=== change + size check ==="
CHANGED="$CHANGED" BASE="$BASE" python3 - <<'PYEOF'
import zipfile, os
a = zipfile.ZipFile(os.environ["BASE"])
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round29_members.zip")
assert set(a.namelist()) == set(b.namelist())
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
want = sorted(set(os.environ["CHANGED"].split()))
print("scenes changed vs base:", diff)
assert diff == want, f"MISMATCH: changed {diff}, intended {want}"
nb = os.path.getsize("/mnt/d/avv/submissions/sub_round29_members.zip")
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB = {nb/1e6:.2f} MB")
print(f"  vs DECIMAL 350,000,000 -> {'FITS' if nb < 350_000_000 else 'OVER'}")
print(f"  vs MiB     367,001,600 -> {'FITS' if nb < 367_001_600 else 'OVER'}")
assert nb < 367_001_600, f"OVER even the MiB cap: {nb:,}"
PYEOF
touch /mnt/d/avv/build_r29.DONE
echo "=== R29 BUILD DONE (base=$(basename $BASE) tower=$TLAM chair=$CLAM bonsai=$BLAM) $(date) ==="

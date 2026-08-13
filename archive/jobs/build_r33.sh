#!/bin/bash
# r33 = r32 (77.6907, BEST GRADED) + three independently-measured changes.
#
# 1. TOWERS -- DEADBAND FIX (per-image, transfers ~1x).  Expected +0.016
#    energy_restore reads png_ens, which is round(mean)*255 = an exact integer per pixel, so the
#    uint8 write computes round(n+d) = n + round(d) and every sub-0.5-LSB correction is thrown
#    away. Shipped towers deliver only 0.13-0.58 LSB. Public harness, same lambda, float chain vs
#    shipped chain: lam=1.0 -0.0021 (1% lost) but lam=0.50 -0.0295 (18%) and lam=0.25 -0.0419
#    (43%) -- the damage is worst at exactly the low amplitude our private pool operates at.
#    Fix = rebuild the SAME 4:1:1 mean in float32 and round ONCE. No re-rendering. Gated by
#    asserting round(float mean) reproduces the shipped png_ens except at exact .5 ties.
#    Towers are the one scene family where the operator is LB-PROVEN beneficial (r28 lam=1.0 beat
#    r27 lam=0 by +0.1005), so delivering more of it is the right direction.
#
# 2. BONSAI -- NEW MODEL + lam=0.  Expected: the whole reason for this round.
#    Per-scene triage on train photos (legal GT): bonsai LPIPS 0.2047 vs 0.075-0.091 for the other
#    six scenes; render/GT radial power 0.42 mid / 0.19 high. It alone carries the 0.4-weighted
#    LPIPS term. Cause: it trained at 30k iters / 5M cap while chair and every tower got 60k / 8M,
#    a scheduling shortcut ("bonsai is only 30k/5M so it costs ~1.2h") that outlived its reason.
#    The 17/07 ladder was still climbing at the top and the cap was BINDING ("Saved 5000000").
#    Retrained at 8M/60k with capD's collapse-safe churn (refine_stop 15k, noise_stop 8k) UNCHANGED.
#    LAMBDA=0: swept on 28 real held-out frames -- 0.00 71.955 / 0.25 71.531 / 1.00 70.760 /
#    2.00 69.650. Restoration is MONOTONICALLY HARMFUL here (bonsai's members disagree because the
#    reconstruction failed, not because averaging lost detail), so r32's bonsai lam=0.25 was a
#    -0.424 scene-pt mistake that only cost ~nothing because the deadband ate 97% of it.
#    *** THEREFORE THE DEADBAND FIX MUST NOT BE APPLIED TO BONSAI. ***
#
# 3. CHAIR -- lambda re-set from the chair eval sweep (clamsweep.log), deadband fix applied ONLY
#    if that sweep says lambda>0 is genuinely beneficial. r32's whole +0.0103 came from chair.
#
# Everything not listed is carried BYTE-VERBATIM from r32.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export PYTHONUNBUFFERED=1
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
DATA=/mnt/d/avv/data/phase1/private_set2
OUT=/mnt/d/avv/r33
FLD=/mnt/d/avv/fields_median_g1_g130
CHAIRFLD=/mnt/d/avv/fields_chair/chair_g1_g130.npy
BONSAI_NEW=/mnt/d/avv/r33_bonsai/prod_s555/test_png
mkdir -p $OUT/tower_ens $OUT/video_ens/chair $OUT/video_ens/bonsai

# filled in from the measurements before this script is run
CHAIR_LAM=${CHAIR_LAM:?export CHAIR_LAM from the chair eval sweep}
CHAIR_DEADBAND=${CHAIR_DEADBAND:-1}      # 1 = use float mean, 0 = keep shipped rounded path

CHAIR_M="/mnt/d/avv/r14/chair_aa42/test_png /mnt/d/avv/r14/chair_aa7/test_png \
/mnt/d/avv/r14/chair_aa13/test_png /mnt/d/avv/r17/chair_ema099_seed42/test_png \
/mnt/d/avv/r17/chair_ema099_seed7/test_png /mnt/d/avv/r17/chair_ema099_seed13/test_png \
/mnt/d/avv/r17/chair_depth_seed42/test_png /mnt/d/avv/r28_members/chair/test_png"

echo "=== 1. TOWERS: float-mean deadband fix, lam=1.0, then the SAME median field ==="
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  R22=/mnt/d/avv/r22/tower_ens/$T/png_ens
  MIP=/mnt/d/avv/r25_mip3d/$T/test_png
  M28=/mnt/d/avv/r28_members/$T/test_png
  for d in $R22 $MIP $M28 /mnt/d/avv/r29/tower_ens/$T/png_ens; do
    [ -d "$d" ] || { echo "!!! $T missing $d -- SKIPPING TOWER (will carry r32 verbatim)"; continue 2; }
  done
  mkdir -p $OUT/tower_ens/$T
  # towers_db.sh may already have produced these; reuse rather than spend another 12 min/scene
  if [ "$(ls $OUT/tower_ens/$T/png/*.png 2>/dev/null | wc -l)" -eq 60 ]; then
    echo "    $T already built (60) -- reusing"; continue
  fi
  # --k 8 kept for continuity with the shipped chain (true effective k is 7.71 under 4:1:1;
  # the difference moves (r-1) by 0.26%, far below the deadband effect being fixed here)
  python $ER --mode apply --k 8 --lam 1.0 \
    --ens_dir /mnt/d/avv/r29/tower_ens/$T/png_ens \
    --member_dirs $MIP $M28 \
    --mean_from_dirs $R22 $MIP $M28 --mean_weights 4 1 1 \
    --out_dir $OUT/tower_ens/$T/png_er
  python gsplat_track/apply_field.py --in_dir $OUT/tower_ens/$T/png_er \
    --field $FLD/${T}.npy --out_dir $OUT/tower_ens/$T/png --strict
  n=$(ls $OUT/tower_ens/$T/png/*.png 2>/dev/null | wc -l)
  [ "$n" -eq 60 ] || { echo "!!! $T produced $n pngs, expected 60"; exit 1; }
  echo "    $T ok ($n)"
done

echo "=== 2. BONSAI: lam=0 (swept: restoration is monotonically HARMFUL on this scene) ==="
# BONSAI_SRC selects what bonsai ships. Default = the plain 6/7-member ensemble with NO restore,
# which is byte-for-byte what r31 shipped and what the lambda sweep says is optimal
# (lam 0.00 -> 71.955, 0.25 -> 71.531, 1.00 -> 70.760 on 28 real held-out frames).
# The retrained cap-8M model only replaces it if it clears 72.31 on the SAME 28 holes
# (72.013 = the shipped 6-member ensemble, + 0.30 single-seed noise floor). That gate is
# evaluated OUTSIDE this script; set BONSAI_SRC explicitly to override.
BONSAI_SRC=${BONSAI_SRC:-/mnt/d/avv/r29/video_ens/bonsai/png_ens}
[ -d "$BONSAI_SRC" ] || { echo "!!! no bonsai source at $BONSAI_SRC"; exit 1; }
nb=$(ls $BONSAI_SRC/*.png 2>/dev/null | wc -l)
[ "$nb" -eq 28 ] || { echo "!!! bonsai source has $nb pngs, expected 28"; exit 1; }
cp $BONSAI_SRC/*.png $OUT/video_ens/bonsai/
echo "    bonsai ok ($nb) from $BONSAI_SRC"

echo "=== 3. CHAIR: lam=$CHAIR_LAM (deadband=$CHAIR_DEADBAND) -> field ==="
if [ "$CHAIR_LAM" = "0" ] || [ "$CHAIR_LAM" = "0.0" ]; then
  echo "    chair lam=0 -> carrying r32 chair bytes verbatim (handled in the assembler)"
  CHAIR_VERBATIM=1
else
  CHAIR_VERBATIM=0
  EXTRA=""
  [ "$CHAIR_DEADBAND" = "1" ] && EXTRA="--mean_from_dirs $CHAIR_M"
  python $ER --mode apply --k 8 --lam $CHAIR_LAM \
    --ens_dir /mnt/d/avv/r29/video_ens/chair/png_ens \
    --member_dirs $CHAIR_M $EXTRA \
    --out_dir $OUT/video_ens/chair/png_er
  python gsplat_track/apply_field.py --in_dir $OUT/video_ens/chair/png_er \
    --field $CHAIRFLD --out_dir $OUT/video_ens/chair/png --strict
  nc=$(ls $OUT/video_ens/chair/png/*.png 2>/dev/null | wc -l)
  [ "$nc" -eq 58 ] || { echo "!!! chair produced $nc pngs, expected 58"; exit 1; }
  echo "    chair ok ($nc)"
fi

echo "=== assemble on r32 (anything not rebuilt is carried byte-verbatim) ==="
CHAIR_VERBATIM=$CHAIR_VERBATIM python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
BASE="/mnt/d/avv/submissions/sub_round32_videolam.zip"
OUT ="/mnt/d/avv/submissions/sub_round33_bonsaicap.zip"
SHIPPED=dict(quality=100, subsampling=2, optimize=True, progressive=True)
SRC={}
for T in ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"]:
    p=f"/mnt/d/avv/r33/tower_ens/{T}/png"
    if os.path.isdir(p) and len(os.listdir(p))==60: SRC[T]=p
SRC["bonsai"]="/mnt/d/avv/r33/video_ens/bonsai"
if os.environ.get("CHAIR_VERBATIM")!="1":
    SRC["chair"]="/mnt/d/avv/r33/video_ens/chair/png"
# Per-scene JPEG quality. r31/r32 shipped HCM0421 at q99 and everything else at q100 -- a pure
# byte-budget concession (verified: re-encoding r31's stored png at q99 reproduces the shipped
# zip bytes 8/8, at q100 it does not). The deadband fix SHRINKS tower files ~0.54% (dithered
# corrections compress better), which together with r32's 15.49 MiB headroom pays for restoring
# HCM0421 to q100 (+6.7 MiB) with ~10 MiB still spare. Higher quality is strictly closer to the
# master, i.e. the opposite direction from the q98/4:4:4 downgrade that LOST on the leaderboard.
CAP=367001600
Q={"HCM0421":100,"HCM0539":100,"HCM0540":100,"HCM0644":100,"HCM0674":100,"chair":100,"bonsai":100}
def build(qmap):
    buf=io.BytesIO(); n=c=0
    src=zipfile.ZipFile(BASE)
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_STORED) as z:
        for i in src.infolist():
            scene=i.filename.split("/")[0]
            if scene not in SRC:
                z.writestr(i.filename, src.read(i.filename)); c+=1; continue
            stem=os.path.splitext(os.path.basename(i.filename))[0]
            p=os.path.join(SRC[scene], stem+".png")
            assert os.path.exists(p), f"missing master: {p}"
            kw=dict(SHIPPED); kw["quality"]=qmap[scene]
            b=io.BytesIO(); Image.open(p).convert("RGB").save(b,"JPEG",**kw)
            z.writestr(i.filename, b.getvalue()); n+=1
    return buf.getvalue(), n, c
data,n,c=build(Q)
if len(data)>CAP:                      # ladder down exactly as previous rounds did
    print(f"  q100 everywhere = {len(data):,} B, OVER CAP -- dropping HCM0421 to q99")
    Q["HCM0421"]=99; data,n,c=build(Q)
    if len(data)>CAP:
        print(f"  still over at {len(data):,} B -- dropping HCM0539 to q99 as well")
        Q["HCM0539"]=99; data,n,c=build(Q)
assert len(data)<=CAP, f"cannot fit under cap: {len(data):,} B"
open(OUT,"wb").write(data)
print(f"re-encoded {n}, carried {c} verbatim  (rebuilt scenes: {sorted(SRC)})")
print(f"quality map shipped: {Q}")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round33_bonsaicap.zip --data_root $DATA
python3 - <<'PYEOF'
import zipfile,os
z=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round33_bonsaicap.zip")
r32=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round32_videolam.zip")
tot=sum(i.file_size for i in z.infolist()); CAP=367001600
changed=sorted({i.filename.split("/")[0] for i in z.infolist()
                if z.read(i.filename)!=r32.read(i.filename)})
print(f"files {len(z.infolist())}/386   {tot:,} B = {tot/2**20:.2f} MiB / cap {CAP/2**20:.0f} MiB "
      f"-> {'FITS' if tot<=CAP else 'OVER CAP'}   headroom {(CAP-tot)/2**20:.2f} MiB")
print("scenes changed vs r32:", changed)
PYEOF
echo "=== r33 BUILT -- NOT SUBMITTED ==="
touch /mnt/d/avv/build_r33.DONE

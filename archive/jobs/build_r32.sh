#!/bin/bash
# r32 = r31 (77.6804, BEST) + video-scene energy restore at a MEASURED lambda.
# Towers, encode and fields IDENTICAL to r31. Only chair and bonsai pixels change.
#
# THE LAST OPEN AXIS, AND IT IS NOW MEASURED RATHER THAN GUESSED (lambda_diag.py, NO GT needed).
# The operator is r = sqrt(1 + (k/(k-1))*V/E) -- driven only by member disagreement V over local
# energy E, both computable without ground truth. Towers are KNOWN to want lam=1.0 (harness sweep:
# 0.75 +0.1756 / 1.0 +0.1883 / 1.25 +0.1820 / 1.5 +0.1570). Measured r-maps:
#     HCM0421 (tower, lam=1.0)  mean r 1.2135  %clamped 0.217  V/E 0.684  boost 21.3%
#     HCM0644 (tower, lam=1.0)  mean r 1.3365  %clamped 0.377  V/E 1.151  boost 33.6%
#     chair   (ships lam=0)     mean r 1.6452  %clamped 1.582  V/E 2.434  boost 64.5%
#     bonsai  (ships lam=0)     mean r 2.1920  %clamped 2.632  V/E 4.763  boost 119.2%
# At lam=1.0 the video scenes would take 2-4x the tower boost with 4-12x the clamp saturation --
# which is exactly why r28's chair 1.25 / bonsai 1.0 went net negative. But lam=0 is the OPPOSITE
# boundary and nothing says the optimum sits there.
# MATCHED-BOOST PRESCRIPTION (target the towers' ~27% optimum):
#     chair  27/64.5  = 0.42 -> ship 0.40
#     bonsai 27/119.2 = 0.23 -> ship 0.25
# ASYMMETRY: towers gain +0.1883 from lam=1.0 vs 0. If the video scenes capture even a third of
# that at matched boost it is +0.018 blended. Downside bounded: r28's FULL-strength lambda cost
# -0.02..-0.03 and this runs at ~40% of it. ~+0.02 up vs ~-0.01 down.
# HONEST CAVEAT: the lam=0 decision itself rests on backing the tower prediction out of a
# three-change bundle (r28->r29), so "lam=0 beats lam=1.0 on video" was never cleanly measured.
# This round tests the interior of an axis whose endpoints are both softly established.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export PYTHONUNBUFFERED=1
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
OUT=/mnt/d/avv/r32
CHAIRFLD=/mnt/d/avv/fields_chair/chair_g1_g130.npy
mkdir -p $OUT/video_ens/chair $OUT/video_ens/bonsai
CHAIR_M="/mnt/d/avv/r14/chair_aa42/test_png /mnt/d/avv/r14/chair_aa7/test_png \
/mnt/d/avv/r14/chair_aa13/test_png /mnt/d/avv/r17/chair_ema099_seed42/test_png \
/mnt/d/avv/r17/chair_ema099_seed7/test_png /mnt/d/avv/r17/chair_ema099_seed13/test_png \
/mnt/d/avv/r17/chair_depth_seed42/test_png /mnt/d/avv/r28_members/chair/test_png"
BONSAI_M="/mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
/mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png \
/mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png \
/mnt/d/avv/r28_members/bonsai/test_png"

echo "=== chair: 8-member mean -> restore(lam=0.40, k=8) -> lens field gauss1 x1.30 ==="
python $ER --mode apply --k 8 --lam 0.40 --ens_dir /mnt/d/avv/r29/video_ens/chair/png_ens \
  --member_dirs $CHAIR_M --out_dir $OUT/video_ens/chair/png_er
python gsplat_track/apply_field.py --in_dir $OUT/video_ens/chair/png_er \
  --field $CHAIRFLD --out_dir $OUT/video_ens/chair/png --strict
[ "$(ls $OUT/video_ens/chair/png/*.png | wc -l)" -eq 58 ] || { echo "!!! chair count"; exit 1; }

echo "=== bonsai: 7-member mean -> restore(lam=0.25, k=7), no field ==="
python $ER --mode apply --k 7 --lam 0.25 --ens_dir /mnt/d/avv/r29/video_ens/bonsai/png_ens \
  --member_dirs $BONSAI_M --out_dir $OUT/video_ens/bonsai/png
[ "$(ls $OUT/video_ens/bonsai/png/*.png | wc -l)" -eq 28 ] || { echo "!!! bonsai count"; exit 1; }

echo "=== assemble on r31 (towers byte-verbatim) ==="
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
BASE="/mnt/d/avv/submissions/sub_round31_fields.zip"
OUT ="/mnt/d/avv/submissions/sub_round32_videolam.zip"
SHIPPED=dict(quality=100, subsampling=2, optimize=True, progressive=True)
SRC={"chair":"/mnt/d/avv/r32/video_ens/chair/png","bonsai":"/mnt/d/avv/r32/video_ens/bonsai/png"}
src=zipfile.ZipFile(BASE); n=c=0
with zipfile.ZipFile(OUT,"w",zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene=i.filename.split("/")[0]
        if scene not in SRC:
            z.writestr(i.filename, src.read(i.filename)); c+=1; continue
        stem=os.path.splitext(os.path.basename(i.filename))[0]
        p=os.path.join(SRC[scene], stem+".png")
        assert os.path.exists(p), f"missing master: {p}"
        b=io.BytesIO(); Image.open(p).convert("RGB").save(b,"JPEG",**SHIPPED)
        z.writestr(i.filename, b.getvalue()); n+=1
print(f"re-encoded {n} (chair+bonsai), carried {c} verbatim (5 towers)")
PYEOF
echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round32_videolam.zip \
  --data_root /mnt/d/avv/data/phase1/private_set2
python3 - <<'PYEOF'
import zipfile, os
a=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round31_fields.zip")
b=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round32_videolam.zip")
assert set(a.namelist())==set(b.namelist())
diff=sorted({n.split('/')[0] for n in a.namelist() if a.read(n)!=b.read(n)})
print("scenes changed vs r31:", diff)
assert diff==["bonsai","chair"], f"expected ONLY the video scenes, got {diff}"
nb=os.path.getsize("/mnt/d/avv/submissions/sub_round32_videolam.zip")
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB   cap -> {'FITS' if nb<367_001_600 else 'OVER'}")
assert nb<367_001_600
PYEOF
touch /mnt/d/avv/build_r32.DONE
echo "=== r32 BUILT -- NOT SUBMITTED ==="

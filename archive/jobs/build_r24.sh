#!/bin/bash
# r24 = r22 with bonsai deepened 3 -> 6 members. MINIMAL-RISK build: towers and chair are copied
# BYTE-IDENTICALLY out of the r22 zip (no re-encode, no re-render), so bonsai is the only thing that
# changes and the comparison against r22's 77.2318 is clean.
#
# WHY bonsai: it was the shallowest ensemble we ship (3, vs chair 7 and towers 6-7), so it sits on
# the steep part of the 1/N curve. Composition is the ONLY lever that has never transferred below 1x
# in 10 graded rounds -- it is variance reduction, not render cleanup, so it does not depend on the
# proxy/production noise gap that made r23 lose.
# Reference marginals from our own graded rounds: tower 2->3 seeds +0.232, 3->4 +0.142 per scene.
#
# ENCODE: the SHIPPED profile (q100 ss2 optimize progressive) -- verified byte-exact against r21's
# bonsai. NOT keep_rgb: r23 proved that loses on production-quality renders.
# PROVENANCE: verified that mean(aa42,aa7,aa13) reproduces the shipped bonsai 28/28 PIXEL-IDENTICAL.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
OUT=/mnt/d/avv/r24
mkdir -p $OUT

echo "=== 6-member bonsai pixel mean ==="
python ensemble_renders.py \
  --dirs /mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
         /mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png \
         /mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png \
  --out $OUT/bonsai/jpg --png_dir $OUT/bonsai/png --names_from $DATA/bonsai/test/test_poses.csv

echo "=== assemble: towers+chair verbatim from r22, bonsai re-encoded at the shipped profile ==="
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
R22 = "/mnt/d/avv/submissions/sub_round22_seed101towers.zip"
OUT = "/mnt/d/avv/submissions/sub_round24_bonsai6.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
src = zipfile.ZipFile(R22)
png = "/mnt/d/avv/r24/bonsai/png"
n_copy = n_enc = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        if i.filename.startswith("bonsai/"):
            stem = os.path.splitext(os.path.basename(i.filename))[0]
            im = Image.open(os.path.join(png, stem + ".png")).convert("RGB")
            b = io.BytesIO(); im.save(b, "JPEG", **SHIPPED)
            z.writestr(i.filename, b.getvalue()); n_enc += 1
        else:
            z.writestr(i.filename, src.read(i.filename)); n_copy += 1
print(f"copied {n_copy} tower/chair files verbatim, re-encoded {n_enc} bonsai files")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round24_bonsai6.zip --data_root $DATA

echo "=== confirm ONLY bonsai differs from r22 ==="
python3 - <<'PYEOF'
import zipfile
a = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round22_seed101towers.zip")
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round24_bonsai6.zip")
assert set(a.namelist()) == set(b.namelist()), "filename sets differ!"
diff = [n for n in a.namelist() if a.read(n) != b.read(n)]
scenes = sorted({n.split("/")[0] for n in diff})
print(f"{len(diff)} files differ from r22, in scenes: {scenes}")
assert scenes == ["bonsai"], f"UNEXPECTED scenes changed: {scenes}"
print("CONFIRMED: bonsai is the only change; towers+chair byte-identical to r22")
PYEOF
touch /mnt/d/avv/build_r24.DONE
echo "=== R24 BUILD DONE $(date) ==="

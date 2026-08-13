#!/bin/bash
# r25 = r24 + a mip3d member on all 5 towers, at the SWEPT weight w_mip = 0.200.
# SINGLE-VARIABLE vs r24: chair + bonsai copied byte-identically; only the 5 tower scenes change.
#
# WEIGHT: the sweep on the HCM0421 proxy (both members have real GT there) put the optimum at
# w_mip = 0.60 in a 2-member blend, i.e. mip3d is worth ~1.50x a normal member => in a 6-normal +
# 1-mip3d ensemble that is 0.200, not the uniform 0.143. Worth +0.0244 over uniform. This matters
# because uniform weighting was only ever shown optimal WITHIN a family; mip3d is a different family
# AND measurably better solo (+0.457 on HCM0421, +0.660 on HCM0181, all three metrics up both times).
#
# COMPOSITION MATH: r22/r24's tower png_ens is the PRE-field 6-member mean. Blending
#   0.8 * old_mean + 0.2 * mip3d   is exactly a 7-way weighted mean with mip3d at 0.200 and each of
# the 6 originals at 0.8/6 = 0.1333. ensemble_renders.py normalises internally, so weights 4 and 1
# give 0.8/0.2.
# Fields: the EXISTING mean-fitted r2r9 fields, unchanged, so r25 isolates the mip3d member.
# (The median-refit field is a separate, independently-validated change -> r26.)
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
OUT=/mnt/d/avv/r25
mkdir -p $OUT/tower_ens

for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  echo "=== $T: 6-member base (w 0.8) + mip3d (w 0.2) -> field -> ==="
  mkdir -p $OUT/tower_ens/$T
  python ensemble_renders.py \
    --dirs /mnt/d/avv/r22/tower_ens/$T/png_ens /mnt/d/avv/r25_mip3d/$T/test_png \
    --weights 4 1 \
    --out $OUT/tower_ens/$T/jpg_tmp --png_dir $OUT/tower_ens/$T/png_ens \
    --names_from $DATA/$T/test/test_poses.csv
  python gsplat_track/apply_field.py --in_dir $OUT/tower_ens/$T/png_ens \
    --field /mnt/d/avv/r2r9/fields/$T.npy --out_dir $OUT/tower_ens/$T/png --strict
done

echo "=== assemble: 5 towers new, chair+bonsai byte-identical from r24 ==="
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
R24 = "/mnt/d/avv/submissions/sub_round24_bonsai6.zip"
OUT = "/mnt/d/avv/submissions/sub_round25_mip3d.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
src = zipfile.ZipFile(R24)
# match r24/r22's per-scene quality so the ONLY change is pixels, not encode
Q = {"HCM0421": 99, "HCM0539": 100, "HCM0540": 100, "HCM0644": 100, "HCM0674": 100}
n_copy = n_enc = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        if scene in Q:
            stem = os.path.splitext(os.path.basename(i.filename))[0]
            p = f"/mnt/d/avv/r25/tower_ens/{scene}/png/{stem}.png"
            im = Image.open(p).convert("RGB")
            kw = dict(SHIPPED); kw["quality"] = Q[scene]
            b = io.BytesIO(); im.save(b, "JPEG", **kw)
            z.writestr(i.filename, b.getvalue()); n_enc += 1
        else:
            z.writestr(i.filename, src.read(i.filename)); n_copy += 1
print(f"re-encoded {n_enc} tower files, copied {n_copy} chair/bonsai files verbatim")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round25_mip3d.zip --data_root $DATA

echo "=== confirm ONLY the 5 towers differ from r24 ==="
python3 - <<'PYEOF'
import zipfile
a = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round24_bonsai6.zip")
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round25_mip3d.zip")
assert set(a.namelist()) == set(b.namelist()), "filename sets differ!"
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
print("scenes changed vs r24:", diff)
assert diff == ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"], f"unexpected: {diff}"
print("CONFIRMED: chair + bonsai byte-identical to r24; only towers changed")
PYEOF
touch /mnt/d/avv/build_r25.DONE
echo "=== R25 BUILD DONE $(date) ==="

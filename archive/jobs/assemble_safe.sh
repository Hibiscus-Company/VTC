#!/bin/bash
# Assemble + verify the decimal-safe r28 from the restore outputs already on disk.
# The build script died at the assembler because one of my heredoc patches never applied (trailing
# space in the search string), so SKIP_SCENES was never set and it went looking for bonsai PNGs that
# the lam=0 branch had deliberately not produced. Rather than redo five towers of restore work,
# bonsai's UNCHANGED source PNGs were staged into the same tree -- they re-encode byte-identically
# to r27's bonsai (verified 28/28 earlier), so the result is exactly what lam=0 means.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2

python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
R27 = "/mnt/d/avv/submissions/sub_round27_lanczos.zip"
OUT = "/mnt/d/avv/submissions/sub_round28_energy_decimalsafe.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q = {"HCM0421": 99, "HCM0539": 100, "HCM0540": 100, "HCM0644": 100, "HCM0674": 100,
     "chair": 100, "bonsai": 100}
SRC = {**{t: f"/mnt/d/avv/r28e_safe/tower_ens/{t}/png" for t in
          ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")},
       "chair": "/mnt/d/avv/r28e_safe/video_ens/chair/png",
       "bonsai": "/mnt/d/avv/r28e_safe/video_ens/bonsai/png"}
src = zipfile.ZipFile(R27)
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        stem = os.path.splitext(os.path.basename(i.filename))[0]
        im = Image.open(os.path.join(SRC[scene], stem + ".png")).convert("RGB")
        kw = dict(SHIPPED); kw["quality"] = Q[scene]
        b = io.BytesIO(); im.save(b, "JPEG", **kw)
        z.writestr(i.filename, b.getvalue()); n += 1
print(f"re-encoded all {n} files")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round28_energy_decimalsafe.zip --data_root $DATA

echo "=== change + size check ==="
python3 - <<'PYEOF'
import zipfile, os, collections
a = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round27_lanczos.zip")
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round28_energy_decimalsafe.zip")
assert set(a.namelist()) == set(b.namelist())
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
print("scenes changed vs r27:", diff)
assert "bonsai" not in diff, "bonsai must be byte-identical at lam=0"
assert len(diff) == 6, f"expected the 5 towers + chair, got {diff}"
nb = os.path.getsize("/mnt/d/avv/submissions/sub_round28_energy_decimalsafe.zip")
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB = {nb/1e6:.2f} MB")
print(f"  vs DECIMAL cap 350,000,000 -> {'FITS' if nb < 350_000_000 else 'OVER'}"
      f" (margin {(350_000_000-nb)/1e6:+.2f} MB)")
print(f"  vs MiB     cap 367,001,600 -> {'FITS' if nb < 367_001_600 else 'OVER'}")
assert nb < 350_000_000, f"decimal-safe build is NOT decimal-safe: {nb:,} bytes"
PYEOF
touch /mnt/d/avv/build_r28safe.DONE
echo "=== DECIMAL-SAFE ASSEMBLED $(date) ==="

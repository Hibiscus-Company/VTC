#!/bin/bash
# jpegalloc SHIP PATH: rebuild the r29 zip with a different JPEG encode, from the PNG MASTERS.
#
# Usage: ja_reencode.sh <quality> <subsampling> <out_zip>
#   e.g. ja_reencode.sh 98 0 /mnt/d/avv/submissions/sub_round30_q98ss0.zip
#
# WHY IT MUST READ THE MASTERS, NOT THE r29 ZIP: verify_zip.py's "single-gen JPEG" check exists
# because re-encoding an already-encoded JPEG costs -0.14..-0.26. build_r29.sh copies unchanged
# scenes VERBATIM out of the base zip, so an encode change cannot be applied by patching the zip
# -- every one of the seven scenes has to come from /mnt/d/avv/r29/**/png again. Those masters are
# the post-energy-restore, post-field PNGs, i.e. exactly the bytes r29's assembler encoded.
#
# The FILE LIST is taken from the r29 zip, not from the master directories: HCM0421's master dir
# holds 61 PNGs while the scene has 60 test poses, and the zip's namelist is the authority the
# scorer matches on.
set -e
Q=${1:?usage: ja_reencode.sh <quality> <subsampling> <out_zip>}
SS=${2:?}; OUT=${3:?}
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1

Q=$Q SS=$SS OUT=$OUT python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
Q, SS, OUT = int(os.environ["Q"]), int(os.environ["SS"]), os.environ["OUT"]
BASE = "/mnt/d/avv/submissions/sub_round29_members.zip"
SRC = {**{t: f"/mnt/d/avv/r29/tower_ens/{t}/png" for t in
          ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")},
       "chair": "/mnt/d/avv/r29/video_ens/chair/png",
       "bonsai": "/mnt/d/avv/r29/video_ens/bonsai/png"}
src = zipfile.ZipFile(BASE)
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        stem = os.path.splitext(os.path.basename(i.filename))[0]
        p = os.path.join(SRC[scene], stem + ".png")
        assert os.path.exists(p), f"missing master {p}"
        im = Image.open(p).convert("RGB")
        b = io.BytesIO()
        im.save(b, "JPEG", quality=Q, subsampling=SS, optimize=True, progressive=True)
        z.writestr(i.filename, b.getvalue())
        n += 1
nb = os.path.getsize(OUT)
print(f"re-encoded {n} files at quality={Q} subsampling={SS}")
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB (cap 350.00 MiB = 367,001,600 B)")
assert nb < 367_001_600, f"OVER CAP: {nb:,}"
PYEOF

python scripts/verify_zip.py --zip "$OUT" --data_root /mnt/d/avv/data/phase1/private_set2
echo "=== DONE $OUT ==="

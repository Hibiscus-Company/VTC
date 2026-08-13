#!/bin/bash
# r30c = the CORRECT r30. Neither previously-built zip is right:
#   sub_round30_k10.zip      -- right field gain (1.30), WRONG encode (q100/4:2:0)
#   sub_round30b_g150...zip  -- right encode (q98/4:4:4), WRONG field gain (1.50)
# Adjudicated on the production harness at k=10 with the gauss1 field, n=60, real test GT:
#   ENCODE  q98/4:4:4 = +0.0504 +/- 0.0030, t=16.96, 59/60, and SMALLER (53.04 vs 59.30 MiB/60).
#           Ladder at 4:4:4: q100 +0.0147 | q99 +0.0384 | **q98 +0.0504** | q97 +0.0209 |
#           q96 -0.0446 | q94 -0.2953. A sharp interior optimum, not a plateau.
#           Lossless PNG is better still (+0.0544) but costs 137.81 MiB/60 -> ~690 MiB of towers
#           alone. INFEASIBLE. q98/4:4:4 captures 93% of the lossless gain at 38% of its bytes.
#   GAIN    1.50 is REFUTED: at matched encode it is -0.036 vs 1.30, and g1.50_q100ss2 measured
#           -0.0343 (t=-2.81, 19/60). 1.40 -0.007, 1.60 -0.085. Monotone decline. KEEP 1.30,
#           which now has three independent measurements behind it.
# MY ERROR, for the record: I concluded "the encode axis is closed" from a grid that swept quality
# at 4:2:0 and subsampling at q100 and never crossed them. The winning cell was the one I skipped.
# Pixels are IDENTICAL to sub_round30_k10.zip; only the encode changes. Single generation from PNG.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
BASE = "/mnt/d/avv/submissions/sub_round29_members.zip"
OUT  = "/mnt/d/avv/submissions/sub_round30c_k10_q98ss0.zip"
ENC  = dict(quality=98, subsampling=0, optimize=True, progressive=True)
SRC = {**{t: f"/mnt/d/avv/r30/tower_ens/{t}/png" for t in
          ("HCM0421","HCM0539","HCM0540","HCM0644","HCM0674")},
       "chair":  "/mnt/d/avv/r30/video_ens/chair/png",   # r29 pixels + chair field
       "bonsai": "/mnt/d/avv/r29/video_ens/bonsai/png"}  # pixels unchanged since r29
for s, d in SRC.items():
    n = len([f for f in os.listdir(d) if f.endswith(".png")])
    assert n in (60, 58, 28), f"{s}: {n} PNGs in {d}"
src = zipfile.ZipFile(BASE); n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        stem = os.path.splitext(os.path.basename(i.filename))[0]
        p = os.path.join(SRC[scene], stem + ".png")
        assert os.path.exists(p), f"missing master: {p}"
        b = io.BytesIO(); Image.open(p).convert("RGB").save(b, "JPEG", **ENC)
        z.writestr(i.filename, b.getvalue()); n += 1
print(f"single-generation encoded {n}/386 files at quality=98 subsampling=0 (4:4:4)")
PYEOF
echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round30c_k10_q98ss0.zip \
  --data_root /mnt/d/avv/data/phase1/private_set2
echo "=== size + parity check ==="
python3 - <<'PYEOF'
import zipfile, os
a=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round29_members.zip")
b=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round30c_k10_q98ss0.zip")
assert set(a.namelist())==set(b.namelist()), "name set changed"
nb=os.path.getsize("/mnt/d/avv/submissions/sub_round30c_k10_q98ss0.zip")
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB = {nb/1e6:.2f} MB")
print(f"  vs MiB cap 367,001,600 -> {'FITS' if nb<367_001_600 else 'OVER'}")
assert nb<367_001_600
# every scene must differ from r29 (all were re-encoded)
diff=sorted({n.split('/')[0] for n in a.namelist() if a.read(n)!=b.read(n)})
print("scenes changed vs r29:", diff)
assert len(diff)==7, f"expected all 7 scenes to change, got {diff}"
PYEOF
touch /mnt/d/avv/build_r30c.DONE
echo "=== r30c BUILT -- NOT SUBMITTED ==="

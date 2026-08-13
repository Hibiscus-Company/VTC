#!/bin/bash
# r30b = the already-built r30 (k=10 towers + gauss1 field + chair field) with TWO more changes:
#        field gain 1.30 -> 1.50 on the five towers, and JPEG q98/4:4:4 on all seven scenes.
#
# MEASURED JOINTLY ON THE r30 CHAIN ITSELF (final_r30.py, HCM0181, 10-member pool,
# restore lam=1 k=10, gauss1 B11-class field, n=60 real test poses, full chain):
#     vs the built k10 zip:   gain 1.45 +0.0414 (t 6.67)   gain 1.50 +0.0467 (t 5.55)
#                             q98/ss0   +0.0507 (t16.89, 58/60, -10.5% bytes)
#                             1.45+q98  +0.0911 (t13.63, 58/60)
#                             1.50+q98  +0.0964 (t11.14, 56/60)   <- SHIPPED
#                             1.60+q98  +0.0951 (t 7.34)          <- turnover, 1.50 is interior
#     additivity ratio 0.99 -- the two changes do not substitute for each other.
#
# WHY 1.50 AND NOT THE SHIPPED 1.30, confirmed twice and once ON THE PRIVATE SET:
#   (a) refute_class.log fit the field twice per PRIVATE tower, once against 30k/5M renders and
#       once against 60k/8M renders: the 30k field is 1.1486x +- 0.0250 the 60k field, corr
#       0.987-0.993, 5/5 towers. The 1.30 constant was calibrated on a 30k/5M model
#       (HCM0181_gsplatB9ut); every private field is fit on <T>_ut42, whose train_args.txt reads
#       `--iters 60000 --cap_max 8000000`. 1.30 x 1.1486 = 1.493.
#   (b) the harness argmax above is 1.50. Two independent routes, 0.5% apart.
#   The gain is an ABSORPTION correction: a more converged model has already swallowed more of the
#   misregistration at train poses, so DIS sees a smaller remainder and the constant must be bigger.
#
# NOT CHANGED: chair keeps gain 1.30 (its field is fit on a different model class that was never
# measured, and chair is exactly where r28's extrapolation went negative). bonsai pixels untouched.
# The energy-restore mu is NOT changed: measured +0.0033 (t=1.43, 33/60) at production depth.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
OUT=/mnt/d/avv/r30b
ZIP=/mnt/d/avv/submissions/sub_round30b_g150_q98ss0.zip
mkdir -p $OUT/tower_ens

# ------------------------------------------------------------------ 1. gauss1 fields at gain 1.50
python - <<'PYEOF'
import numpy as np, os, json, glob
SRC, DST = "/mnt/d/avv/fields_median_g1_g130", "/mnt/d/avv/fields_median_g1_g150"
os.makedirs(DST, exist_ok=True)
for p in sorted(glob.glob(SRC + "/*.npy")):
    b = os.path.basename(p)
    f = (np.load(p).astype(np.float32) * (1.50 / 1.30))
    assert np.abs(f).max() < 8.0, f"{b} max {np.abs(f).max():.2f}px trips apply_field"
    np.save(os.path.join(DST, b), f)
    m = json.load(open(p + ".meta.json"))
    assert os.sep + "test" + os.sep not in m.get("gt_dir", "") + os.sep, "Rule 10 violation"
    m["scale"] = 1.50
    m["rationale"] = (m.get("rationale", "") + " | r30b: 1.30 -> 1.50. The 1.30 constant was fit "
        "against a 30k/5M model; the private fields are fit on <T>_ut42 (60k/8M) and the 30k field "
        "measures 1.1486x the 60k field on all five private towers -> 1.493. Harness argmax on the "
        "k=10 chain is 1.50 (+0.0467 alone, +0.0964 with q98/ss0, n=60 real test GT).")
    json.dump(m, open(os.path.join(DST, b + ".meta.json"), "w"), indent=1)
    print(f"{b}: mean|f| {np.abs(f).mean():.4f}  max|f| {np.abs(f).max():.4f}")
PYEOF

# ------------------------------------------------------------------ 2. re-warp the r30 png_er
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  [ "$(ls /mnt/d/avv/r30/tower_ens/$T/png_er/*.png 2>/dev/null | wc -l)" -eq 60 ] \
    || { echo "!!! $T: r30 png_er is not 60 PNGs -- ABORT"; exit 1; }
  rm -rf $OUT/tower_ens/$T/png
  mkdir -p $OUT/tower_ens/$T
  echo "=== $T: re-warp r30 png_er with gauss1 field x1.50 ==="
  python gsplat_track/apply_field.py --in_dir /mnt/d/avv/r30/tower_ens/$T/png_er \
    --field /mnt/d/avv/fields_median_g1_g150/$T.npy --out_dir $OUT/tower_ens/$T/png --strict
  [ "$(ls $OUT/tower_ens/$T/png/*.png 2>/dev/null | wc -l)" -eq 60 ] \
    || { echo "!!! $T: warp output count wrong"; exit 1; }
done

# ------------------------------------------------------------------ 3. single-generation q98/4:4:4
# File list from the r29 zip (the scorer's authority -- master dirs carry an extra
# field_applied.json and HCM0421 once had a 61st PNG). Every scene re-encodes from a PNG master,
# so nothing is a second JPEG generation.
ZIP=$ZIP python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
OUT = os.environ["ZIP"]
BASE = "/mnt/d/avv/submissions/sub_round29_members.zip"
SRC = {**{t: f"/mnt/d/avv/r30b/tower_ens/{t}/png" for t in
          ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")},
       "chair": "/mnt/d/avv/r30/video_ens/chair/png",       # k10 build: r29 pixels + chair field
       "bonsai": "/mnt/d/avv/r29/video_ens/bonsai/png"}     # pixels unchanged since r29
ENC = dict(quality=98, subsampling=0, optimize=True, progressive=True)
src = zipfile.ZipFile(BASE)
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        stem = os.path.splitext(os.path.basename(i.filename))[0]
        p = os.path.join(SRC[scene], stem + ".png")
        assert os.path.exists(p), f"missing master {p}"
        b = io.BytesIO()
        Image.open(p).convert("RGB").save(b, "JPEG", **ENC)
        z.writestr(i.filename, b.getvalue()); n += 1
nb = os.path.getsize(OUT)
print(f"encoded {n} files at quality=98 subsampling=0 (4:4:4)")
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB  (cap 350.00 MiB = 367,001,600 B)")
assert nb < 367_001_600, f"OVER CAP: {nb:,}"
PYEOF

python scripts/verify_zip.py --zip "$ZIP" --data_root $DATA

ZIP=$ZIP python3 - <<'PYEOF'
import zipfile, os
from PIL import Image
import io
a = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round30_k10.zip")
b = zipfile.ZipFile(os.environ["ZIP"])
assert set(a.namelist()) == set(b.namelist()), "namelist drift vs the k10 zip"
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
print("scenes changed vs sub_round30_k10.zip:", diff)
assert len(diff) == 7, f"expected all 7 to change (encode moved), got {diff}"
# confirm 4:4:4 and progressive actually landed
for n in list(b.namelist())[:3] + list(b.namelist())[-3:]:
    im = Image.open(io.BytesIO(b.read(n)))
    sf = {c[1] for c in im.layer}
    print(f"  {n}: sampling {sorted(sf)} progressive={im.info.get('progressive',0)}")
    assert sf == {(1, 1)}, f"{n} is not 4:4:4"
PYEOF
touch /mnt/d/avv/build_r30b.DONE
echo "=== r30b BUILT -- NOT SUBMITTED -- $(date) ==="

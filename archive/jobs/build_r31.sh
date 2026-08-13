#!/bin/bash
# r31 = r29 (77.6644, our BEST) + ONLY the two per-image FIELD operators.
# Reverts both changes that graded NEGATIVE, isolated by submitting r30 and r30c:
#     r30  - r29  = 2 extra members + gauss1 field + chair field = -0.0030
#     r30c - r30  = encode q98/4:4:4, alone                      = -0.0019
# WHY THE MEMBERS GO BACK: 8->10 gave PSNR +0.0073 and LPIPS WORSE by +0.0309pp -- exactly the
# harness's PAST-THE-OPTIMUM signature (at k=12 harness PSNR still climbed 26.2245->26.2338 while
# LPIPS turned 0.09348->0.09501). Our pool's heterogeneity is 2.20/255 vs the harness pool's
# 4.67/255, so homogeneous members carry less independent information and our k=8 is ALREADY as
# over-smoothed as a diverse pool at k~12. EFFECTIVE k != NOMINAL k. r29's k=8 was right.
# WHY THE ENCODE GOES BACK: q98/4:4:4 cross-validated on single-member renders of all 5 public
# towers is -0.0094 mean, 0/5 positive, NEGATIVE ON HCM0181 ITSELF -- the scene whose k=10 ensemble
# gave +0.0504. The gain belonged to ensemble over-smoothing, not to the encode. LB said -0.0019.
#
# WHAT SURVIVES, AND IT IS CROSS-VALIDATED THIS TIME (gauss_xscene.log, single member, real GT):
#   gauss(sigma=1) field smoothing vs plain, gain 1.30:
#     HCM0181 +0.0095 (60/60) | HCM0193 +0.0082 (56/60) | HCM0204 +0.0073 (57/60)
#     hcm0031 +0.0080 (47/50) | hcm0034 +0.0077 (55/60)
#     5-SCENE MEAN +0.0081, POSITIVE 5/5, tight 0.0073-0.0095 band.
#   sigma=2 is nominally +0.0091 but with a wider spread and weaker win rates (40-53 vs 47-60);
#   the +0.0010 is negligible blended, so sigma=1 wins on consistency.
#   chair lens field: LOVO +0.0299, a per-image operator in the transferable class. NOT
#   cross-validatable (no chair test GT). Kept, and flagged as the one unverified item.
# EXPECTED: +0.0058 (gauss1 over 5/7 scenes) + ~0.004 (chair) = **+0.010** -> ~77.674.
# Deliberately modest. Everything here is per-image and pool-independent; nothing in r31 depends on
# ensemble diversity, which is what broke r30 and r30c.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export PYTHONUNBUFFERED=1
OUT=/mnt/d/avv/r31; FLD=/mnt/d/avv/fields_median_g1_g130
mkdir -p $OUT/tower_ens
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  [ "$(ls /mnt/d/avv/r29/tower_ens/$T/png_er/*.png 2>/dev/null | wc -l)" -eq 60 ] \
    || { echo "!!! $T: r29 png_er not 60 -- ABORT"; exit 1; }
  echo "=== $T: r29 png_er (k=8, lam=1.0) -> gauss1 x1.30 field ==="
  mkdir -p $OUT/tower_ens/$T
  python gsplat_track/apply_field.py --in_dir /mnt/d/avv/r29/tower_ens/$T/png_er \
    --field $FLD/$T.npy --out_dir $OUT/tower_ens/$T/png --strict
  [ "$(ls $OUT/tower_ens/$T/png/*.png 2>/dev/null | wc -l)" -eq 60 ] || { echo "!!! $T count"; exit 1; }
done
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
BASE="/mnt/d/avv/submissions/sub_round29_members.zip"
OUT ="/mnt/d/avv/submissions/sub_round31_fields.zip"
SHIPPED=dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q={"HCM0421":99,"HCM0539":100,"HCM0540":100,"HCM0644":100,"HCM0674":100,"chair":100,"bonsai":100}
SRC={**{t:f"/mnt/d/avv/r31/tower_ens/{t}/png" for t in
        ("HCM0421","HCM0539","HCM0540","HCM0644","HCM0674")},
     "chair":"/mnt/d/avv/r30/video_ens/chair/png"}   # r29 chair pixels + chair field
src=zipfile.ZipFile(BASE); n=c=0
with zipfile.ZipFile(OUT,"w",zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene=i.filename.split("/")[0]
        if scene=="bonsai":                      # untouched since r29 -> carry bytes verbatim
            z.writestr(i.filename, src.read(i.filename)); c+=1; continue
        stem=os.path.splitext(os.path.basename(i.filename))[0]
        p=os.path.join(SRC[scene], stem+".png")
        assert os.path.exists(p), f"missing master: {p}"
        kw=dict(SHIPPED); kw["quality"]=Q[scene]
        b=io.BytesIO(); Image.open(p).convert("RGB").save(b,"JPEG",**kw)
        z.writestr(i.filename, b.getvalue()); n+=1
print(f"re-encoded {n}, carried {c} verbatim (bonsai)")
PYEOF
echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round31_fields.zip \
  --data_root /mnt/d/avv/data/phase1/private_set2
python3 - <<'PYEOF'
import zipfile, os
a=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round29_members.zip")
b=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round31_fields.zip")
assert set(a.namelist())==set(b.namelist())
diff=sorted({n.split('/')[0] for n in a.namelist() if a.read(n)!=b.read(n)})
print("scenes changed vs r29:", diff)
assert diff==sorted(["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair"]), diff
nb=os.path.getsize("/mnt/d/avv/submissions/sub_round31_fields.zip")
print(f"TOTAL {nb:,} bytes = {nb/1048576:.2f} MiB   cap 367,001,600 -> {'FITS' if nb<367_001_600 else 'OVER'}")
assert nb<367_001_600
PYEOF
touch /mnt/d/avv/build_r31.DONE
echo "=== r31 BUILT -- NOT SUBMITTED ==="

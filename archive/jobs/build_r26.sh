#!/bin/bash
# r26 = r25 with the lens fields REFIT using a per-pixel MEDIAN instead of the arithmetic mean.
# SINGLE-VARIABLE vs r25: identical pre-field ensembles (r25/tower_ens/<T>/png_ens, which already
# include the mip3d member at w=0.2), identical encode, chair+bonsai byte-identical. ONLY the field
# changes.
#
# WHY: fit_field.py pools cv2 DIS optical flow with a per-pixel arithmetic MEAN. DIS returns ~0
# across the 15-22% sky (no texture to track) and blows up at occlusion boundaries; a mean swallows
# both failure modes, a median rejects them.
# VALIDATED leave-one-view-out (each view never contributes to the field applied to it), with the
# PROJECT scorer including LPIPS:
#     HCM0421  mean 81.9527 -> median 82.0151   +0.0625
#     HCM0539  mean 82.5538 -> median 82.6620   +0.1082
# LPIPS improved slightly too (.0657->.0653, .0647->.0642), so it is not a fidelity-for-perception
# trade. GT-STRUCTURE-MATCHING class -- the one class with LB-confirmed ~1x transfer (the field
# itself: predicted +0.88, delivered +0.73). Zero GPU training.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
FLD=/mnt/d/avv/fields_median
mkdir -p $FLD

for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  echo "=== $T: refitting field with median estimator ==="
  python gsplat_track/fit_field.py \
    --render_dir /mnt/d/avv/r2r9/models/${T}_ut42/train_png \
    --gt_dir $DATA/$T/train/images \
    --out $FLD/$T.npy --estimator median
done

echo ""
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  echo "=== $T: applying median field to r25's ensemble ==="
  rm -rf /mnt/d/avv/r26/tower_ens/$T/png
  mkdir -p /mnt/d/avv/r26/tower_ens/$T
  python gsplat_track/apply_field.py --in_dir /mnt/d/avv/r25/tower_ens/$T/png_ens \
    --field $FLD/$T.npy --out_dir /mnt/d/avv/r26/tower_ens/$T/png --strict
done

echo "=== assemble: 5 towers refielded, chair+bonsai byte-identical from r25 ==="
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
R25 = "/mnt/d/avv/submissions/sub_round25_mip3d.zip"
OUT = "/mnt/d/avv/submissions/sub_round26_medianfield.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q = {"HCM0421": 99, "HCM0539": 100, "HCM0540": 100, "HCM0644": 100, "HCM0674": 100}
src = zipfile.ZipFile(R25)
n_enc = n_copy = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        if scene in Q:
            stem = os.path.splitext(os.path.basename(i.filename))[0]
            im = Image.open(f"/mnt/d/avv/r26/tower_ens/{scene}/png/{stem}.png").convert("RGB")
            kw = dict(SHIPPED); kw["quality"] = Q[scene]
            b = io.BytesIO(); im.save(b, "JPEG", **kw)
            z.writestr(i.filename, b.getvalue()); n_enc += 1
        else:
            z.writestr(i.filename, src.read(i.filename)); n_copy += 1
print(f"re-encoded {n_enc} tower files, copied {n_copy} chair/bonsai verbatim")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round26_medianfield.zip --data_root $DATA

echo "=== confirm ONLY the towers differ from r25 ==="
python3 - <<'PYEOF'
import zipfile
a = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round25_mip3d.zip")
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round26_medianfield.zip")
assert set(a.namelist()) == set(b.namelist())
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
print("scenes changed vs r25:", diff)
assert diff == ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"], f"unexpected: {diff}"
print("CONFIRMED: single-variable (field estimator only); chair+bonsai byte-identical")
PYEOF
touch /mnt/d/avv/build_r26.DONE
echo "=== R26 BUILD DONE $(date) ==="

#!/bin/bash
# r27 = r26 with ONE flag changed: the lens-field warp resamples with INTER_LANCZOS4
# instead of INTER_CUBIC.
#
# WHY: the warp is the last operator that touches every shipped pixel, and every resample
# destroys high-frequency energy permanently. Round-trip isolation (warp by +f then -f, no
# GT): cubic keeps 0.948 of the Laplacian energy, lanczos4 keeps 0.976 -- cubic was throwing
# away 5.2% of the detail, lanczos4 only 2.4%.
# MEASURED against REAL test GT on the production harness (5 public towers, 290 images,
# models trained on 100% of their train photos, fields fit on TRAIN photos only):
#     as PNG                            +0.1095, 5/5 scenes
#     after the shipped q100/ss2 JPEG   +0.1078, 5/5 scenes
# PSNR, SSIM and LPIPS ALL improve in every scene -- pure fidelity, not a perception trade.
# Independently corroborated by private-set leave-one-view-out (+0.0980, 150 held-out views).
#
# SINGLE-VARIABLE vs r26: same median fields (byte-identical .npy, NOT refit), same input
# ensembles (r25/tower_ens/<T>/png_ens), same per-scene encode, chair+bonsai byte-identical.
# Zero GPU.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
FLD=/mnt/d/avv/fields_median

echo "=== assert the lanczos flag is actually live ==="
python3 -c "
import sys, inspect, cv2
sys.path.insert(0,'gsplat_track')
import fit_field
d = inspect.signature(fit_field.apply_field).parameters['interp'].default
assert d == cv2.INTER_LANCZOS4, f'apply_field default interp is {d}, expected INTER_LANCZOS4 ({cv2.INTER_LANCZOS4})'
print('OK: apply_field defaults to INTER_LANCZOS4')
"

for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  echo "=== $T: re-applying the SAME median field with lanczos4 ==="
  rm -rf /mnt/d/avv/r27/tower_ens/$T/png
  mkdir -p /mnt/d/avv/r27/tower_ens/$T
  python gsplat_track/apply_field.py --in_dir /mnt/d/avv/r25/tower_ens/$T/png_ens \
    --field $FLD/$T.npy --out_dir /mnt/d/avv/r27/tower_ens/$T/png --strict
done

echo "=== confirm the fields are the very ones r26 shipped (no silent refit) ==="
python3 - <<'PYEOF'
import json, os
for T in ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"]:
    m = json.load(open(f"/mnt/d/avv/r27/tower_ens/{T}/png/field_applied.json"))
    fm = m["field_meta"]
    assert "/test/" not in fm["gt_dir"] + "/", fm["gt_dir"]
    print(f"  {T}: field {os.path.basename(m['field'])} fit on {fm['gt_dir']}  n={m['n']}")
PYEOF

echo "=== assemble: 5 towers re-warped, chair+bonsai byte-identical from r26 ==="
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
R26 = "/mnt/d/avv/submissions/sub_round26_medianfield.zip"
OUT = "/mnt/d/avv/submissions/sub_round27_lanczos.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q = {"HCM0421": 99, "HCM0539": 100, "HCM0540": 100, "HCM0644": 100, "HCM0674": 100}
src = zipfile.ZipFile(R26)
n_enc = n_copy = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        scene = i.filename.split("/")[0]
        if scene in Q:
            stem = os.path.splitext(os.path.basename(i.filename))[0]
            im = Image.open(f"/mnt/d/avv/r27/tower_ens/{scene}/png/{stem}.png").convert("RGB")
            kw = dict(SHIPPED); kw["quality"] = Q[scene]
            b = io.BytesIO(); im.save(b, "JPEG", **kw)
            z.writestr(i.filename, b.getvalue()); n_enc += 1
        else:
            z.writestr(i.filename, src.read(i.filename)); n_copy += 1
print(f"re-encoded {n_enc} tower files, copied {n_copy} chair/bonsai verbatim")
PYEOF

echo "=== verify ==="
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round27_lanczos.zip --data_root $DATA

echo "=== confirm ONLY the towers differ from r26 ==="
python3 - <<'PYEOF'
import zipfile, os
a = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round26_medianfield.zip")
b = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round27_lanczos.zip")
assert set(a.namelist()) == set(b.namelist())
diff = sorted({n.split("/")[0] for n in a.namelist() if a.read(n) != b.read(n)})
print("scenes changed vs r26:", diff)
assert diff == ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"], f"unexpected: {diff}"
sz = os.path.getsize("/mnt/d/avv/submissions/sub_round27_lanczos.zip") / 1e6
print(f"CONFIRMED: single-variable (remap kernel only); chair+bonsai byte-identical. {sz:.1f} MB")
assert sz < 350, f"OVER BUDGET: {sz:.1f} MB"
PYEOF
touch /mnt/d/avv/build_r27.DONE
echo "=== R27 BUILD DONE $(date) ==="

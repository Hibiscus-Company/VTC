#!/bin/bash
# Finish r22: JPEG-encode the 5 fresh field-corrected tower PNGs (single-generation),
# then merge in chair+bonsai copied VERBATIM (byte-identical, zero re-encode) from r21's
# zip, then verify_zip.py. NEVER auto-submit.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
conda activate fastgs2
DATA=/mnt/d/avv/data/phase1/private_set2
TOWERS_ZIP=/mnt/d/avv/r22/towers_only.zip
FINAL=/mnt/d/avv/submissions/sub_round22_seed101towers.zip
R21=/mnt/d/avv/submissions/sub_round21_chairdepth_hcm0674ema.zip

echo "=== encoding 5 towers (budget 300MB, chair+bonsai fixed 47.4MB copied verbatim) ==="
python build_submission_zip.py \
  --scene_dirs HCM0421=/mnt/d/avv/r22/tower_ens/HCM0421/png \
               HCM0539=/mnt/d/avv/r22/tower_ens/HCM0539/png \
               HCM0540=/mnt/d/avv/r22/tower_ens/HCM0540/png \
               HCM0644=/mnt/d/avv/r22/tower_ens/HCM0644/png \
               HCM0674=/mnt/d/avv/r22/tower_ens/HCM0674/png \
  --data_root $DATA --out $TOWERS_ZIP --max_mb 300

echo "=== merging: fresh towers + chair/bonsai copied verbatim from r21 (no re-encode) ==="
python3 - <<'PYEOF'
import zipfile
towers = zipfile.ZipFile("/mnt/d/avv/r22/towers_only.zip")
r21 = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round21_chairdepth_hcm0674ema.zip")
with zipfile.ZipFile("/mnt/d/avv/submissions/sub_round22_seed101towers.zip", "w", zipfile.ZIP_STORED) as out:
    for i in towers.infolist():
        out.writestr(i.filename, towers.read(i.filename))
    n_cb = 0
    for i in r21.infolist():
        if i.filename.startswith(("chair/", "bonsai/")):
            out.writestr(i.filename, r21.read(i.filename))
            n_cb += 1
    print(f"copied {n_cb} chair/bonsai files verbatim from r21")
PYEOF

echo "=== verify_zip.py ==="
python scripts/verify_zip.py --zip $FINAL --data_root $DATA

echo "=== size check ==="
python3 -c "import os; print(f'{os.path.getsize(\"$FINAL\")/1e6:.1f} MB')"
touch /mnt/d/avv/build_r22_final.DONE
echo "=== R22 BUILD DONE $(date) ==="

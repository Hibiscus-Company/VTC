#!/bin/bash
# r23 = r22's EXACT pixels, re-encoded with the fixed JPEG profile. Zero model change, zero GPU.
# Video scenes (where the measured tax lives) get subsampling=0 + keep_rgb=True, which removes the
# RGB->YCbCr roundtrip: bonsai +1.01, chair +0.13. Towers are immune (tax ~0.006) so they stay on
# the cheap path and absorb the byte cost via the quality ladder.
# Sources are all BYTE-VERIFIED against r21/r22 (see EXPERIMENTS.md 26/07 provenance entry).
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
OUT=/mnt/d/avv/submissions/sub_round23b_jpegfix.zip

python build_submission_zip.py \
  --scene_dirs HCM0421=/mnt/d/avv/r22/tower_ens/HCM0421/png \
               HCM0539=/mnt/d/avv/r22/tower_ens/HCM0539/png \
               HCM0540=/mnt/d/avv/r22/tower_ens/HCM0540/png \
               HCM0644=/mnt/d/avv/r22/tower_ens/HCM0644/png \
               HCM0674=/mnt/d/avv/r22/tower_ens/HCM0674/png \
               chair=/mnt/d/avv/r21/video_ens/chair7/png \
               bonsai=/mnt/d/avv/r16/video_ens/bonsai/png \
  --data_root $DATA --out $OUT \
  --qualities 100 99 98 97 96 95 94 93 92 --subsampling 2 \
  --hq_quality 98 \
  --hq_scenes chair bonsai --hq_min_quality 96 --max_mb 350

echo "=== verify ==="
python scripts/verify_zip.py --zip $OUT --data_root $DATA

echo "=== confirm video scenes really are keep_rgb RGB-mode jpegs, towers unchanged ==="
python3 - <<'PYEOF'
import zipfile, io
from PIL import Image
z = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round23b_jpegfix.zip")
r22 = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round22_seed101towers.zip")
for scene in ("bonsai", "chair", "HCM0421"):
    n = sorted(x for x in z.namelist() if x.startswith(scene + "/"))[0]
    b = z.read(n)
    im = Image.open(io.BytesIO(b))
    # a keep_rgb jpeg reports 3 components with RGB component ids
    adobe = b.find(b'Adobe')
    print(f"{scene:8s} {n:28s} fmt={im.format} mode={im.mode} size={im.size} "
          f"bytes={len(b):8d}  (r22 {len(r22.read(n)):8d})")
    assert im.format == "JPEG", f"{scene} is not a JPEG!"
    assert im.size == Image.open(io.BytesIO(r22.read(n))).size
print("all scenes decode as real JPEGs at correct size")
PYEOF
touch /mnt/d/avv/build_r23.DONE
echo "=== R23 BUILD DONE $(date) ==="

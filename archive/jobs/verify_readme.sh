#!/bin/bash
# End-to-end check of REPRODUCE_r36.md: run the bonsai path EXACTLY as the document writes it,
# using only repo-relative paths, and compare the result byte-for-byte with the shipped zip.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
DATA=/mnt/d/avv/data/phase1/private_set2
V=/mnt/d/avv/verify_readme; rm -rf $V; mkdir -p $V
M="/mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png /mnt/d/avv/r14/bonsai_aa13/test_png \
/mnt/d/avv/r24_bonsai/aa101/test_png /mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png \
/mnt/d/avv/r28_members/bonsai/test_png"
NEW="/mnt/d/avv/r38_prod/s111/test_png /mnt/d/avv/r38_prod/s555/test_png /mnt/d/avv/r38_prod/s777/test_png \
/mnt/d/avv/r38_prod/s222/test_png /mnt/d/avv/r38_prod/s333/test_png /mnt/d/avv/r38_prod/s999/test_png"
echo "--- stage 2: ensemble (README section 6) ---"
python ensemble_renders.py --dirs $M $NEW --out $V/jpg_tmp --png_dir $V/png_ens \
  --names_from $DATA/bonsai/test/test_poses.csv
echo "--- stage 3: restore, k=13 lam=0.25 (README section 6 table) ---"
python energy_restore.py --mode apply --k 13 --lam 0.25 \
  --ens_dir $V/png_ens --member_dirs $NEW --out_dir $V/png_er
echo "--- stage 4: NO field for bonsai (README section 7) ---"
echo "--- assemble + byte-compare (README section 8) ---"
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ENCODE = dict(quality=100, subsampling=2, optimize=True, progressive=True)
z = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round36_bonsai13.zip")
same = diff = 0; bad = []
for i in z.infolist():
    if not i.filename.startswith("bonsai/"): continue
    stem = os.path.splitext(os.path.basename(i.filename))[0]
    b = io.BytesIO()
    Image.open(f"/mnt/d/avv/verify_readme/png_er/{stem}.png").convert("RGB").save(b, "JPEG", **ENCODE)
    if b.getvalue() == z.read(i.filename): same += 1
    else: diff += 1; bad.append(i.filename)
print(f"RESULT  bonsai/: {same} byte-identical, {diff} differ")
if bad: print("  mismatches:", bad[:3])
PYEOF
touch /mnt/d/avv/verify_readme.DONE

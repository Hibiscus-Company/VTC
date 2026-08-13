#!/bin/bash
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
V=/mnt/d/avv/verify_r36
NEW="/mnt/d/avv/r38_prod/s111/test_png /mnt/d/avv/r38_prod/s555/test_png /mnt/d/avv/r38_prod/s777/test_png \
/mnt/d/avv/r38_prod/s222/test_png /mnt/d/avv/r38_prod/s333/test_png /mnt/d/avv/r38_prod/s999/test_png"
python $ER --mode apply --k 13 --lam 0.25 --ens_dir $V/png_ens --member_dirs $NEW --out_dir $V/png
python3 - <<'PYEOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
S=dict(quality=100,subsampling=2,optimize=True,progressive=True)
z=zipfile.ZipFile("/mnt/d/avv/submissions/sub_round36_bonsai13.zip")
same=diff=0; bad=[]
for i in z.infolist():
    if not i.filename.startswith("bonsai/"): continue
    stem=os.path.splitext(os.path.basename(i.filename))[0]
    b=io.BytesIO(); Image.open(f"/mnt/d/avv/verify_r36/png/{stem}.png").convert("RGB").save(b,"JPEG",**S)
    if b.getvalue()==z.read(i.filename): same+=1
    else: diff+=1; bad.append(i.filename)
print(f"REBUILD vs SHIPPED sub_round36_bonsai13.zip  bonsai/: {same} byte-identical, {diff} differ")
if bad: print("  mismatches:", bad[:3])
PYEOF
touch /mnt/d/avv/verify_r36.DONE

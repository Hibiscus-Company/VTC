#!/bin/bash
# r35 = r32 (77.6907, BEST) with bonsai rebuilt as 7 EXISTING members + 3 NEW scale_reg=0.1
# members. Towers and chair carried BYTE-VERBATIM.
#
# EVIDENCE (all on the 28 held-out bonsai holes, today's scorer, ship encode q100/ss2):
#   diversity-matched k=2 gate: ctrl2 (lam0.01 A/A) 71.4996 -> treat2 (lam0.1 A/A) 71.6649
#                               = +0.1653, t=+2.12, 17/28.  Pre-registered gate was >=+0.12, t>=2.0 -> PASS
#   single-model replicates:    lam0.1 {71.9911, 71.9816, 71.8254} vs lam0.01 {71.9030, 71.6951}
#   composition sweep:          5 old 71.9441 -> 5 old + 3 new 72.2527  (+0.309, the best of six mixes)
#   run-to-run noise floor:     0.407 for a single 1-vs-1 A/B, so only the k=2 paired test counts
# bonsai lambda stays 0.25: r33a set it to 0 and LOST -0.0041 on the leaderboard.
# No lens field on bonsai (LOVO n=40: every variant <= no-field).
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
OUT=/mnt/d/avv/r35; mkdir -p $OUT/bonsai
OLD="/mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
/mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png \
/mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png \
/mnt/d/avv/r28_members/bonsai/test_png"
NEW="/mnt/d/avv/r38_prod/s111/test_png /mnt/d/avv/r38_prod/s555/test_png /mnt/d/avv/r38_prod/s777/test_png"
echo "=== bonsai: 10-member mean (7 old + 3 new lam=0.1) ==="
python ensemble_renders.py --dirs $OLD $NEW --out $OUT/bonsai/jpg_tmp --png_dir $OUT/bonsai/png_ens
n=$(ls $OUT/bonsai/png_ens/*.png|wc -l); [ "$n" -eq 28 ] || { echo "!!! ens count $n"; exit 1; }
echo "=== restore lam=0.25, k=10 ==="
python $ER --mode apply --k 10 --lam 0.25 --ens_dir $OUT/bonsai/png_ens \
  --member_dirs $NEW --out_dir $OUT/bonsai/png
n=$(ls $OUT/bonsai/png/*.png|wc -l); [ "$n" -eq 28 ] || { echo "!!! restore count $n"; exit 1; }
echo "=== assemble on r32 (towers + chair byte-verbatim) ==="
python3 - <<'PYEOF'
import zipfile,io,os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
BASE="/mnt/d/avv/submissions/sub_round32_videolam.zip"
OUT ="/mnt/d/avv/submissions/sub_round35_bonsai_scalereg.zip"
S=dict(quality=100,subsampling=2,optimize=True,progressive=True); CAP=367001600
SRC={"bonsai":"/mnt/d/avv/r35/bonsai/png"}
src=zipfile.ZipFile(BASE); buf=io.BytesIO(); n=c=0
with zipfile.ZipFile(buf,"w",zipfile.ZIP_STORED) as z:
    for i in src.infolist():
        sc=i.filename.split("/")[0]
        if sc not in SRC: z.writestr(i.filename,src.read(i.filename)); c+=1; continue
        stem=os.path.splitext(os.path.basename(i.filename))[0]
        p=f"{SRC[sc]}/{stem}.png"; assert os.path.exists(p), p
        b=io.BytesIO(); Image.open(p).convert("RGB").save(b,"JPEG",**S)
        z.writestr(i.filename,b.getvalue()); n+=1
data=buf.getvalue(); assert len(data)<=CAP, f"OVER CAP {len(data):,}"
open(OUT,"wb").write(data)
z=zipfile.ZipFile(OUT); r32=zipfile.ZipFile(BASE)
ch=sorted({i.filename.split('/')[0] for i in z.infolist() if z.read(i.filename)!=r32.read(i.filename)})
print(f"re-encoded {n} (bonsai), carried {c} verbatim")
print(f"files {len(z.infolist())}/386  {len(data):,} B = {len(data)/2**20:.2f} MiB  headroom {(CAP-len(data))/2**20:.2f} MiB")
print(f"scenes changed vs r32: {ch}")
print(f"CRC: {z.testzip() or 'OK'}")
PYEOF
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round35_bonsai_scalereg.zip \
  --data_root /mnt/d/avv/data/phase1/private_set2 2>&1 | tail -12
echo "=== r35 BUILT -- NOT SUBMITTED ==="; touch /mnt/d/avv/build_r35.DONE

#!/bin/bash
# r36 = r35 (77.7106, BEST) with bonsai going from 10 members to 13: the SAME 7 old members plus
# all SIX scale_reg=0.1 production members instead of three. Towers and chair BYTE-VERBATIM again.
#
# This is a pure ADD, not a replacement, and that distinction is load-bearing:
#   - r20 established ADD-not-REPLACE on the LB (+0.0948, 36x the single-slot swap nulls)
#   - the eval mixsweep says pure-new is WORSE than mixed (3 new only 72.0247 < 5 old + 3 new
#     72.2527), so the old members still carry signal and must not be dropped
#   - at fixed 3 new, the sweep is monotone increasing in member count:
#       2 old + 3 new (k=5) 72.2032 < 3 old + 3 new (k=6) 72.2265 < 5 old + 3 new (k=8) 72.2527
#     so adding members in this range has never cost anything on the holes
#
# WHAT WE ARE NOT CLAIMING: there is no eval-split measurement of 6-new compositions, because the
# eval side only ever got 3 lam=0.1 arms (sr01/sr01b are the same seed 42, sr01_s101 is seed 101).
# The mixsweep spread across all five mixes that contain the new members is only 0.15, so the
# signal there is "include them", not "this exact ratio". Expected LB gain is accordingly small,
# roughly +0.005..+0.015, and it is a direction the leaderboard has already paid for once.
#
# lam stays 0.25 (r33a shipped 0 and LOST -0.0041). k must equal the TRUE member count -- the
# ledger caught a k=8-passed-for-9 mis-specification on HCM0421, so: 13 members -> --k 13.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
OUT=/mnt/d/avv/r36; mkdir -p $OUT/bonsai
OLD="/mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
/mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png \
/mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png \
/mnt/d/avv/r28_members/bonsai/test_png"
NEW="/mnt/d/avv/r38_prod/s111/test_png /mnt/d/avv/r38_prod/s555/test_png /mnt/d/avv/r38_prod/s777/test_png \
/mnt/d/avv/r38_prod/s222/test_png /mnt/d/avv/r38_prod/s333/test_png /mnt/d/avv/r38_prod/s999/test_png"
for d in $OLD $NEW; do n=$(ls $d/*.png 2>/dev/null|wc -l); [ "$n" -eq 28 ] || { echo "!!! $d has $n pngs"; exit 1; }; done
echo "=== bonsai: 13-member mean (7 old + 6 new lam=0.1) ==="
python ensemble_renders.py --dirs $OLD $NEW --out $OUT/bonsai/jpg_tmp --png_dir $OUT/bonsai/png_ens
n=$(ls $OUT/bonsai/png_ens/*.png|wc -l); [ "$n" -eq 28 ] || { echo "!!! ens count $n"; exit 1; }
echo "=== restore lam=0.25, k=13 (true member count) ==="
python $ER --mode apply --k 13 --lam 0.25 --ens_dir $OUT/bonsai/png_ens \
  --member_dirs $NEW --out_dir $OUT/bonsai/png
n=$(ls $OUT/bonsai/png/*.png|wc -l); [ "$n" -eq 28 ] || { echo "!!! restore count $n"; exit 1; }
echo "=== assemble on r35 (towers + chair byte-verbatim) ==="
python3 - <<'PYEOF'
import zipfile,io,os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
BASE="/mnt/d/avv/submissions/sub_round35_bonsai_scalereg.zip"
OUT ="/mnt/d/avv/submissions/sub_round36_bonsai13.zip"
S=dict(quality=100,subsampling=2,optimize=True,progressive=True); CAP=367001600
SRC={"bonsai":"/mnt/d/avv/r36/bonsai/png"}
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
z=zipfile.ZipFile(OUT); r35=zipfile.ZipFile(BASE)
ch=sorted({i.filename.split('/')[0] for i in z.infolist() if z.read(i.filename)!=r35.read(i.filename)})
print(f"re-encoded {n} (bonsai), carried {c} verbatim")
print(f"files {len(z.infolist())}/386  {len(data):,} B = {len(data)/2**20:.2f} MiB  headroom {(CAP-len(data))/2**20:.2f} MiB")
print(f"scenes changed vs r35: {ch}")
print(f"CRC: {z.testzip() or 'OK'}")
PYEOF
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round36_bonsai13.zip \
  --data_root /mnt/d/avv/data/phase1/private_set2 2>&1 | tail -12
echo "=== r36 BUILT -- NOT SUBMITTED ==="; touch /mnt/d/avv/build_r36.DONE

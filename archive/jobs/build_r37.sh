#!/bin/bash
# r37 = r36 (77.7230, BEST) with bonsai going 13 -> 17 members: the SAME 13 plus the four new
# long45k production members. Towers and chair BYTE-VERBATIM.
#
# WHY long45k: the pre-registered encoded k=2 paired gate PASSED --
#     ref 71.7394 -> long45k 72.6437, paired +0.9043, sd 2.0617, t=+2.321, 16/28 wins
#     forecast scene +0.7687 -> LB TOTAL +0.1098  (r36 was +0.0124, r35 +0.0199)
# Two seeds agree on the raw split (+0.7117 / +0.8063) and the length curve is unimodal with 45k
# at its peak (30k +0.322, 45k +0.759, 50k +0.523, 60k +0.257).
#
# WHY KEEP ALL 13 OLD MEMBERS even though they measure 0.76 BELOW long45k: measured, not argued.
# The 31/07 encoded composition sweep on the 28 holes:
#     3 old only              72.0247      2 new only              72.6437
#     1 old + 2 new           73.0522      2 old + 2 new           73.0984
#     3 old + 2 new           73.0613      3 old + 2 mid + 2 new   73.1034  <- best
# Every mix containing old members beats pure-new. ADD-not-REPLACE survives a 0.76 quality gap,
# so the ~0.15 member-quality rule does not apply at this range. The top five mixes span only
# 0.055, so the exact ratio barely matters -- which is why no GPU time was spent tuning it.
#
# lam stays 0.25 (r33a shipped lam=0 and LOST -0.0041 on the LB). k MUST equal the true member
# count -- the ledger caught a k=8-passed-for-9 mis-specification once.
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
ER=/home/bkai/.claude/jobs/1c9cf7e9/tmp/energy_restore.py
OUT=/mnt/d/avv/r37; mkdir -p $OUT/bonsai
# wait for the last two production members
while [ ! -f /mnt/d/avv/r45_prod/s202.DONE ] || [ ! -f /mnt/d/avv/r45_prod/s303.DONE ]; do sleep 60; done
echo "=== all 4 long45k production members present $(date +%H:%M) ==="
OLD="/mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png \
/mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png \
/mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png \
/mnt/d/avv/r28_members/bonsai/test_png /mnt/d/avv/r38_prod/s111/test_png \
/mnt/d/avv/r38_prod/s555/test_png /mnt/d/avv/r38_prod/s777/test_png \
/mnt/d/avv/r38_prod/s222/test_png /mnt/d/avv/r38_prod/s333/test_png \
/mnt/d/avv/r38_prod/s999/test_png"
NEW="/mnt/d/avv/r45_prod/s42/test_png /mnt/d/avv/r45_prod/s101/test_png \
/mnt/d/avv/r45_prod/s202/test_png /mnt/d/avv/r45_prod/s303/test_png"
for d in $OLD $NEW; do n=$(ls $d/*.png 2>/dev/null|wc -l); [ "$n" -eq 28 ] || { echo "!!! $d has $n pngs"; exit 1; }; done
echo "=== bonsai: 17-member mean (13 carried + 4 long45k) ==="
python ensemble_renders.py --dirs $OLD $NEW --out $OUT/bonsai/jpg_tmp --png_dir $OUT/bonsai/png_ens
n=$(ls $OUT/bonsai/png_ens/*.png|wc -l); [ "$n" -eq 28 ] || { echo "!!! ens count $n"; exit 1; }
echo "=== restore lam=0.25, k=17 (true member count) ==="
python $ER --mode apply --k 17 --lam 0.25 --ens_dir $OUT/bonsai/png_ens \
  --member_dirs $NEW --out_dir $OUT/bonsai/png
n=$(ls $OUT/bonsai/png/*.png|wc -l); [ "$n" -eq 28 ] || { echo "!!! restore count $n"; exit 1; }
echo "=== assemble on r36 (towers + chair byte-verbatim) ==="
python3 - <<'PYEOF'
import zipfile,io,os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
BASE="/mnt/d/avv/submissions/sub_round36_bonsai13.zip"
OUT ="/mnt/d/avv/submissions/sub_round37_long45k.zip"
S=dict(quality=100,subsampling=2,optimize=True,progressive=True); CAP=367001600
SRC={"bonsai":"/mnt/d/avv/r37/bonsai/png"}
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
z=zipfile.ZipFile(OUT); r36=zipfile.ZipFile(BASE)
ch=sorted({i.filename.split('/')[0] for i in z.infolist() if z.read(i.filename)!=r36.read(i.filename)})
print(f"re-encoded {n} (bonsai), carried {c} verbatim")
print(f"files {len(z.infolist())}/386  {len(data):,} B = {len(data)/2**20:.2f} MiB  headroom {(CAP-len(data))/2**20:.2f} MiB")
print(f"scenes changed vs r36: {ch}")
print(f"CRC: {z.testzip() or 'OK'}")
PYEOF
python scripts/verify_zip.py --zip /mnt/d/avv/submissions/sub_round37_long45k.zip \
  --data_root /mnt/d/avv/data/phase1/private_set2 2>&1 | tail -12
echo "=== r37 BUILT -- NOT SUBMITTED $(date +%H:%M) ==="; touch /mnt/d/avv/build_r37.DONE

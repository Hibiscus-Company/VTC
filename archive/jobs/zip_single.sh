#!/bin/bash
set -e
python3 - <<'PYEOF'
import zipfile, os, time
D="/home/bkai/delivery_r36"
OUT="/home/bkai/VAR2026_BTS_postverification_r36.zip"
t0=time.time(); n=0; DEFL=(".md",".txt",".tsv",".yml",".py",".sh",".json",".csv")
with zipfile.ZipFile(OUT,"w",zipfile.ZIP_STORED,allowZip64=True) as z:
    for root,_,files in os.walk(D):
        for f in sorted(files):
            if f in ("stage.log","STAGED.DONE"): continue
            p=os.path.join(root,f); rel=os.path.join("VAR2026_BTS_postverification_r36",os.path.relpath(p,D))
            ct=zipfile.ZIP_DEFLATED if f.lower().endswith(DEFL) else zipfile.ZIP_STORED
            z.write(p,rel,compress_type=ct); n+=1
s=os.path.getsize(OUT)
print(f"{os.path.basename(OUT)}: {n} files, {s:,} B = {s/2**30:.2f} GiB, {time.time()-t0:.0f}s", flush=True)
PYEOF
python3 - <<'PYEOF'
import zipfile
f="/home/bkai/VAR2026_BTS_postverification_r36.zip"
z=zipfile.ZipFile(f); names={i.filename for i in z.infolist()}
R="VAR2026_BTS_postverification_r36/"
need=[R+x for x in ("README.md","REPRODUCE.md","WEIGHTS_MANIFEST.md","environment/gsplat.yml",
     "environment/fastgs2.yml","code/gsplat_track/train_gsplat.py","code/gsplat_track/render_gsplat.py",
     "code/energy_restore.py","weights/bonsai_r14_aa42.pt","submission/sub_round36_bonsai13.zip")]
print("CRC:", z.testzip() or "OK", "| entries:", len(names))
for x in need: print(("  OK   " if x in names else "  MISS "), x.replace(R,""))
PYEOF
sha256sum /home/bkai/VAR2026_BTS_postverification_r36.zip
touch /home/bkai/SINGLE_ZIP.DONE

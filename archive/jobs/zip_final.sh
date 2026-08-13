#!/bin/bash
set -e
python3 - <<'PYEOF'
import zipfile, os, time
D="/home/bkai/delivery_r36"
OUT="/home/bkai/VAR2026_BTS_postverification.zip"
ROOT="VAR2026_BTS_postverification"
t0=time.time(); n=0; DEFL=(".md",".txt",".tsv",".yml",".py",".sh",".json",".csv")
with zipfile.ZipFile(OUT,"w",zipfile.ZIP_STORED,allowZip64=True) as z:
    for root,_,files in os.walk(D):
        for f in sorted(files):
            if f in ("stage.log","STAGED.DONE"): continue
            p=os.path.join(root,f); rel=os.path.join(ROOT,os.path.relpath(p,D))
            ct=zipfile.ZIP_DEFLATED if f.lower().endswith(DEFL) else zipfile.ZIP_STORED
            z.write(p,rel,compress_type=ct); n+=1
s=os.path.getsize(OUT)
print(f"{os.path.basename(OUT)}: {n} files, {s:,} B = {s/2**30:.2f} GiB, {time.time()-t0:.0f}s", flush=True)
PYEOF
python3 - <<'PYEOF'
import zipfile
f="/home/bkai/VAR2026_BTS_postverification.zip"; R="VAR2026_BTS_postverification/"
z=zipfile.ZipFile(f); names={i.filename for i in z.infolist()}
print("CRC:", z.testzip() or "OK", "| entries:", len(names), flush=True)
need=["SUBMISSIONS.md","README.md","REPRODUCE.md","WEIGHTS_MANIFEST.md","MANIFEST.tsv","SHA256SUMS.txt",
      "environment/gsplat.yml","environment/fastgs2.yml","code/gsplat_track/train_gsplat.py",
      "code/gsplat_track/render_gsplat.py","code/ensemble_renders.py","code/energy_restore.py","code/lapfuse.py",
      "weights/bonsai_long45k_s202.pt","weights/bonsai_long45k_s303.pt","weights/bonsai_r14_aa42.pt",
      "submission/sub_round36_bonsai13.zip","submission/sub_round37_long45k.zip",
      "renders/bonsai/long45k_s42/test_png/frame_000010.png"]
bad=[x for x in need if R+x not in names]
for x in need: print(("  OK   " if R+x in names else "  MISS "), x, flush=True)
print("ALL PRESENT" if not bad else f"MISSING {len(bad)}", flush=True)
PYEOF
sha256sum /home/bkai/VAR2026_BTS_postverification.zip
touch /home/bkai/FINAL_ZIP.DONE

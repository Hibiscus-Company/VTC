#!/bin/bash
# weights (.pt) compress to 93%, PNG to 100% -- DEFLATE buys nothing and costs hours. ZIP_STORED.
set -e
python3 - <<'PYEOF'
import zipfile, os, time
D="/home/bkai/delivery_r36"
for sub, out in (("weights","/home/bkai/r36_weights.zip"), ("renders","/home/bkai/r36_renders.zip")):
    t0=time.time(); n=0
    with zipfile.ZipFile(out,"w",zipfile.ZIP_STORED,allowZip64=True) as z:
        for root,_,files in os.walk(os.path.join(D,sub)):
            for f in files:
                p=os.path.join(root,f); z.write(p, os.path.relpath(p,D)); n+=1
    s=os.path.getsize(out)
    print(f"{os.path.basename(out)}: {n} files, {s:,} B = {s/2**30:.2f} GiB, {time.time()-t0:.0f}s", flush=True)
PYEOF
python3 - <<'PYEOF'
import zipfile
for f in ("/home/bkai/r36_weights.zip","/home/bkai/r36_renders.zip"):
    z=zipfile.ZipFile(f); print(f"{f}: CRC {z.testzip() or 'OK'}, {len(z.infolist())} entries", flush=True)
PYEOF
touch /home/bkai/ZIP_HEAVY.DONE

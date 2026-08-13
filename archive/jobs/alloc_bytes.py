"""What does each encode upgrade COST in bytes, per scene, on the ACTUAL r29 pixels?
Pure CPU, no GT. Extrapolates a sample to the full scene by the sample's own ss2 bytes,
so the estimate is anchored to the true shipped totals."""
import io, os, sys, zipfile, collections
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SRC = {**{t: f"/mnt/d/avv/r29/tower_ens/{t}/png" for t in
          ("HCM0421","HCM0539","HCM0540","HCM0644","HCM0674")},
       "chair": "/mnt/d/avv/r29/video_ens/chair/png",
       "bonsai": "/mnt/d/avv/r29/video_ens/bonsai/png"}
Q29 = {"HCM0421":99,"HCM0539":100,"HCM0540":100,"HCM0644":100,"HCM0674":100,
       "chair":100,"bonsai":100}
ARMS = [("q100_ss2",100,2),("q100_ss0",100,0),("q100_ss1",100,1),("q99_ss0",99,0)]
BASEK = dict(optimize=True, progressive=True)
NS = 10

# true shipped bytes per scene, from the graded artefact
z = zipfile.ZipFile("/mnt/d/avv/submissions/sub_round29_members.zip")
ship = collections.Counter(); nimg = collections.Counter()
for i in z.infolist():
    if i.is_dir(): continue
    s = i.filename.split("/")[0]; ship[s] += i.compress_size; nimg[s] += 1

print(f"{'scene':9s} {'n':>3s} {'shipped MiB':>11s} " + " ".join(f"{a[0]:>10s}" for a in ARMS))
tot = {a[0]: 0.0 for a in ARMS}
for sc, d in SRC.items():
    stems = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    stems = stems[:: max(1, len(stems)//NS)][:NS]
    b = {a[0]: 0 for a in ARMS}
    for f in stems:
        im = Image.open(os.path.join(d, f)).convert("RGB")
        for nm, q, ss in ARMS:
            o = io.BytesIO(); im.save(o, "JPEG", quality=q, subsampling=ss, **BASEK)
            b[nm] += o.tell()
    # anchor: the sample's own r29-setting bytes vs the scene's true shipped bytes
    anchor = f"q{Q29[sc]}_ss2"
    ref = b.get(anchor)
    if ref is None:                       # HCM0421 ships q99/ss2, not in ARMS
        r = 0
        for f in stems:
            im = Image.open(os.path.join(d, f)).convert("RGB")
            o = io.BytesIO(); im.save(o,"JPEG",quality=Q29[sc],subsampling=2,**BASEK); r += o.tell()
        ref = r
    k = ship[sc] / ref
    row = f"{sc:9s} {nimg[sc]:3d} {ship[sc]/1048576:11.2f} "
    for nm,_,_ in ARMS:
        mib = b[nm]*k/1048576; tot[nm] += mib
        row += f"{mib:10.2f}"
    print(row)
print(f"{'TOTAL':9s} {sum(nimg.values()):3d} {sum(ship.values())/1048576:11.2f} " +
      "".join(f"{tot[a[0]]:10.2f}" for a in ARMS))
print(f"\ncap 350 MiB; r29 = {sum(ship.values())/1048576:.2f} MiB; headroom {350-sum(ship.values())/1048576:.2f} MiB")

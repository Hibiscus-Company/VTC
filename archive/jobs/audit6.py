import zipfile, io, numpy as np, os, json
from PIL import Image
R31="/mnt/d/avv/submissions/sub_round31_fields.zip"; R32="/mnt/d/avv/submissions/sub_round32_videolam.zip"
a=zipfile.ZipFile(R31); b=zipfile.ZipFile(R32)
def dj(z,n): return np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"),np.float32)
def dp(p): return np.asarray(Image.open(p).convert("RGB"),np.float32)
def down(x):
    h,w=x.shape[:2]; h-=h%2; w-=w%2; x=x[:h,:w]
    return 0.25*(x[0::2,0::2]+x[1::2,0::2]+x[0::2,1::2]+x[1::2,1::2])
def L0(x):
    d=down(x); H,W=d.shape[0]*2,d.shape[1]*2
    u=np.repeat(np.repeat(d,2,0),2,1)[:H,:W]
    return x[:H,:W]-u
for sc,src in [("chair","/mnt/d/avv/r32/video_ens/chair/png"),("bonsai","/mnt/d/avv/r32/video_ens/bonsai/png")]:
    ns=sorted(n for n in a.namelist() if n.startswith(sc+"/"))[:12]
    al=[];cc=[];jp=[];pl=[];pmad=[]
    for n in ns:
        x=dj(a,n); y=dj(b,n)
        gx=x.mean(2); gy=y.mean(2)
        d=(gy-gx); L=L0(gx); d=d[:L.shape[0],:L.shape[1]]
        al.append(float((d*L).sum()/(L*L).sum()))
        cc.append(float(np.corrcoef(d.ravel(),L.ravel())[0,1]))
        base=os.path.splitext(n.split('/')[1])[0]+".png"
        p=os.path.join(src,base)
        if os.path.exists(p):
            z=dp(p)
            jp.append(float(np.abs(z-y).mean()))          # PNG source vs r32 JPEG
            pmad.append(float(np.abs(z-x).mean()))        # PNG source vs r31 JPEG
            gz=z.mean(2); Lz=L0(gz)
            pl.append(float((Lz**2).mean()/ (L**2).mean()))
    print(f"== {sc} (n={len(ns)})")
    print(f"   effective finest-band amplitude boost alpha = {np.mean(al):+.4f}  (per-img {min(al):+.3f}..{max(al):+.3f})")
    print(f"   corr(delta, L0(r31)) = {np.mean(cc):+.3f}")
    if jp:
        print(f"   |r32_zip - r32_source_png| = {np.mean(jp):.4f} levels   |r31_zip - r32_source_png| = {np.mean(pmad):.4f}")
        print(f"   L0 energy ratio  source_png / r31_zip = {np.mean(pl):.4f}")
    else:
        print("   source png not matched")
p="/mnt/d/avv/r32/video_ens/chair/field_applied.json"
if os.path.exists(p): print("\nfield_applied.json:", open(p).read()[:400])

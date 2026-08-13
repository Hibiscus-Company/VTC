import os,sys,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
D="/mnt/d/avv/data/phase1/private_set2"
SC=[("bonsai","/mnt/d/avv/blurbound/bonsai/train_png"),("chair","/mnt/d/avv/blurbound/chair/train_png"),
    ("HCM0421","/mnt/d/avv/r2r9/models/HCM0421_ut42/train_png")]
g=lambda p: np.asarray(Image.open(p).convert("L"),dtype=np.float32)/255.
def radprof(x):
    F=np.abs(np.fft.fftshift(np.fft.fft2(x-x.mean())))**2
    h,w=x.shape; cy,cx=h//2,w//2
    Y,X=np.ogrid[:h,:w]; R=np.sqrt((Y-cy)**2+(X-cx)**2).astype(int)
    nb=min(cy,cx); tb=np.bincount(R.ravel(),F.ravel(),nb); cnt=np.bincount(R.ravel(),None,nb)
    return tb/np.maximum(cnt,1), nb
print(f"{'scene':>8} {'band':>6} {'render/GT power ratio':>22}")
for name,rd in SC:
    gtd=f"{D}/{name}/train/images"
    gtm={os.path.splitext(f)[0]:os.path.join(gtd,f) for f in os.listdir(gtd)}
    st=sorted(s[:-4] for s in os.listdir(rd) if s.endswith(".png") and s[:-4] in gtm)[:6]
    rr=[];gg=[]
    for s in st:
        a=g(os.path.join(rd,s+".png")); b=g(gtm[s])
        if a.shape!=b.shape: continue
        pa,nb=radprof(a); pb,_=radprof(b); rr.append(pa); gg.append(pb)
    if not rr: print(name,"shape mismatch"); continue
    pa=np.mean(rr,0); pb=np.mean(gg,0); n=len(pa)
    for lo,hi,lab in [(0.0,0.1,"low"),(0.1,0.3,"mid"),(0.3,0.6,"high"),(0.6,1.0,"vhigh")]:
        i0,i1=int(lo*n),max(int(hi*n),int(lo*n)+1)
        print(f"{name:>8} {lab:>6} {pa[i0:i1].sum()/max(pb[i0:i1].sum(),1e-12):22.4f}")

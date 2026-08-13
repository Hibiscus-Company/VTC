import zipfile, io, numpy as np
from PIL import Image
R31="/mnt/d/avv/submissions/sub_round31_fields.zip"
R32="/mnt/d/avv/submissions/sub_round32_videolam.zip"
a=zipfile.ZipFile(R31); b=zipfile.ZipFile(R32)

def dec(z,n):
    return np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"),dtype=np.float32)

def down(x):
    # 2x box downsample on HxWx3
    h,w=x.shape[:2]; h-=h%2; w-=w%2
    x=x[:h,:w]
    return 0.25*(x[0::2,0::2]+x[1::2,0::2]+x[0::2,1::2]+x[1::2,1::2])
def up(x,shape):
    y=np.repeat(np.repeat(x,2,axis=0),2,axis=1)
    return y[:shape[0],:shape[1]]
def bands(x,L=3):
    out=[]
    cur=x
    for i in range(L):
        d=down(cur)
        r=cur[:d.shape[0]*2,:d.shape[1]*2]-up(d,(d.shape[0]*2,d.shape[1]*2))
        out.append(float((r**2).mean()))
        cur=d
    return out

for sc,names in [("chair",None),("bonsai",None)]:
    ns=sorted(n for n in a.namelist() if n.startswith(sc+"/"))
    mads=[];maxs=[];rmses=[];p999=[]
    E31=np.zeros(3);E32=np.zeros(3); nimg=0
    frac_gt1=[];frac_gt4=[]
    signed=[]
    for n in ns:
        x=dec(a,n); y=dec(b,n)
        assert x.shape==y.shape, (n,x.shape,y.shape)
        d=np.abs(y-x)
        mads.append(d.mean()); maxs.append(d.max()); rmses.append(float(np.sqrt(((y-x)**2).mean())))
        p999.append(float(np.percentile(d,99.9)))
        frac_gt1.append(float((d>1).mean())); frac_gt4.append(float((d>4).mean()))
        signed.append(float((y-x).mean()))
        gx=x.mean(2); gy=y.mean(2)
        E31+=np.array(bands(gx)); E32+=np.array(bands(gy)); nimg+=1
    E31/=nimg;E32/=nimg
    print(f"== {sc}  n={len(ns)}")
    print(f"   mean|diff| (8-bit levels) = {np.mean(mads):.4f}   per-image min/max = {min(mads):.4f}/{max(mads):.4f}")
    print(f"   RMS diff = {np.mean(rmses):.4f}    mean signed (r32-r31) = {np.mean(signed):+.4f}")
    print(f"   max|diff| over images = {max(maxs):.1f}   mean 99.9pct|diff| = {np.mean(p999):.2f}")
    print(f"   frac pixels |d|>1 = {100*np.mean(frac_gt1):.2f}%   |d|>4 = {100*np.mean(frac_gt4):.2f}%")
    print(f"   PSNR(r32 vs r31) = {10*np.log10(255**2/np.mean([r**2 for r in rmses])):.2f} dB")
    print(f"   Laplacian band energy (luma, mean over imgs):")
    for i in range(3):
        print(f"     L{i} (finest={i==0}): r31={E31[i]:.4f}  r32={E32[i]:.4f}  ratio r32/r31={E32[i]/E31[i]:.4f}")
    print()

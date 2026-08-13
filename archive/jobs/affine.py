import os,sys,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
gtd="/mnt/d/avv/evalsplit/bonsai/eval_gt"
gt={os.path.splitext(f)[0]:os.path.join(gtd,f) for f in os.listdir(gtd)}
for tag,rd in [("pC","/mnt/d/avv/bonsai_perc/pC_lpearly/eval_png"),
               ("dp05","/mnt/d/avv/depth/bonsai_dp05/eval_png")]:
    p0=[];p1=[];p2=[];gains=[]
    for f in sorted(os.listdir(rd)):
        s=os.path.splitext(f)[0]
        if s not in gt: continue
        r=np.asarray(Image.open(os.path.join(rd,f)).convert("RGB"),dtype=np.float64)/255.
        g=np.asarray(Image.open(gt[s]).convert("RGB"),dtype=np.float64)/255.
        p0.append(10*np.log10(1/max(((r-g)**2).mean(),1e-12)))
        # per-channel gain only (pure exposure)
        rg=r.copy()
        for c in range(3):
            a=(r[...,c]*g[...,c]).sum()/max((r[...,c]**2).sum(),1e-12); rg[...,c]=r[...,c]*a
            if c==1: gains.append(a)
        p1.append(10*np.log10(1/max(((np.clip(rg,0,1)-g)**2).mean(),1e-12)))
        # per-channel gain+bias
        ra=r.copy()
        for c in range(3):
            x=r[...,c].ravel();y=g[...,c].ravel()
            A=np.vstack([x,np.ones_like(x)]).T
            co,*_=np.linalg.lstsq(A,y,rcond=None); ra[...,c]=co[0]*r[...,c]+co[1]
        p2.append(10*np.log10(1/max(((np.clip(ra,0,1)-g)**2).mean(),1e-12)))
    p0,p1,p2=np.array(p0),np.array(p1),np.array(p2)
    print(f"{tag:5s} n={len(p0)} PSNR raw {p0.mean():.4f} | gain-only {p1.mean():.4f} (+{p1.mean()-p0.mean():.4f}) | gain+bias {p2.mean():.4f} (+{p2.mean()-p0.mean():.4f})")
    print(f"      per-image G gain: mean {np.mean(gains):.4f} std {np.std(gains):.4f} min {min(gains):.4f} max {max(gains):.4f}")

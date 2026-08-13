import os, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
def ps(a,b): return 10*np.log10(255.0**2/np.mean((a-b)**2))
for sc,gtd,rd in [('HCM0421','/mnt/d/avv/evalsplit/HCM0421/eval_gt','/mnt/d/avv/evalgen/HCM0421/eval_png'),
                  ('HCM0421_mip','/mnt/d/avv/evalsplit/HCM0421/eval_gt','/mnt/d/avv/mip3d/HCM0421_mip0.2/eval_png'),
                  ('chair','/mnt/d/avv/evalsplit/chair/eval_gt','/mnt/d/avv/chair_eval/base60k/eval_png')]:
    if not os.path.isdir(rd): continue
    r0=[];r1=[];r2=[];off=[]
    for g in sorted(os.listdir(gtd)):
        r=os.path.join(rd,os.path.splitext(g)[0]+'.png')
        if not os.path.exists(r): continue
        A=np.asarray(Image.open(os.path.join(gtd,g)).convert('RGB'),dtype=np.float64)
        B=np.asarray(Image.open(r).convert('RGB'),dtype=np.float64)
        if A.shape!=B.shape: continue
        d=(A-B).reshape(-1,3).mean(0); off.append(d)
        r0.append(ps(A,B)); r1.append(ps(A,np.clip(B+d,0,255)))
        # per-frame per-channel gain+bias oracle
        Bc=B.copy()
        for c in range(3):
            X=np.c_[B[...,c].ravel(),np.ones(B[...,c].size)]
            k,b=np.linalg.lstsq(X,A[...,c].ravel(),rcond=None)[0]
            Bc[...,c]=k*B[...,c]+b
        r2.append(ps(A,np.clip(Bc,0,255)))
    off=np.array(off)
    print(f"{sc}: n={len(r0)} PSNR raw {np.mean(r0):.3f} -> +oracle DC {np.mean(r1):.3f} (+{np.mean(r1)-np.mean(r0):.3f}) -> +oracle per-ch gain/bias {np.mean(r2):.3f} (+{np.mean(r2)-np.mean(r0):.3f})")
    print(f"   per-frame DC offset: mean {off.mean(0).round(2)}  std {off.std(0).round(2)}  max|.| {np.abs(off).max():.1f}")

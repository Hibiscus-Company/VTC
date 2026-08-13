import os, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
cfg={'chair':('/mnt/d/avv/evalsplit/chair/eval_gt','/mnt/d/avv/chair_eval/base60k/eval_png'),
     'bonsai':('/mnt/d/avv/evalsplit/bonsai/eval_gt','/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png'),
     'HCM0421':('/mnt/d/avv/evalsplit/HCM0421/eval_gt','/mnt/d/avv/evalgen/HCM0421/eval_png')}
for sc,(gtd,rd) in cfg.items():
    if not os.path.isdir(rd): print('miss',sc,rd); continue
    fr=[]; tot=0; hot=[]
    for g in sorted(os.listdir(gtd)):
        r=os.path.join(rd,os.path.splitext(g)[0]+'.png')
        if not os.path.exists(r): continue
        A=np.asarray(Image.open(os.path.join(gtd,g)).convert('RGB'),dtype=np.float32)
        B=np.asarray(Image.open(r).convert('RGB'),dtype=np.float32)
        if A.shape!=B.shape: continue
        e=((A-B)**2).mean(2).ravel()
        s=np.sort(e)[::-1]; cs=np.cumsum(s)/e.sum()
        n=len(e)
        fr.append([cs[int(0.01*n)],cs[int(0.05*n)],cs[int(0.10*n)],cs[int(0.25*n)]])
        # per-channel mean offset (WB/exposure of the render vs GT)
        hot.append((A-B).reshape(-1,3).mean(0))
    f=np.array(fr).mean(0); h=np.array(hot)
    print(f"== {sc} n={len(fr)}: share of squared error in the worst  1% px {f[0]:.2f} | 5% {f[1]:.2f} | 10% {f[2]:.2f} | 25% {f[3]:.2f}")
    print(f"   per-frame mean RGB offset (GT-render): mean {h.mean(0).round(2)} levels, per-frame std {h.std(0).round(2)}, |offset| mean {np.abs(h).mean():.2f}")

import os, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
for sc,gtd,rd in [('HCM0421','/mnt/d/avv/evalsplit/HCM0421/eval_gt','/mnt/d/avv/evalgen/HCM0421/eval_png'),
                  ('chair','/mnt/d/avv/evalsplit/chair/eval_gt','/mnt/d/avv/chair_eval/base60k/eval_png'),
                  ('bonsai','/mnt/d/avv/evalsplit/bonsai/eval_gt','/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png')]:
    r=[]
    for g in sorted(os.listdir(gtd)):
        p=os.path.join(rd,os.path.splitext(g)[0]+'.png')
        if not os.path.exists(p): continue
        A=np.asarray(Image.open(os.path.join(gtd,g)).convert('RGB'),dtype=np.float64)
        B=np.asarray(Image.open(p).convert('RGB'),dtype=np.float64)
        d=(A-B).reshape(-1,3).mean(0)
        r.append((g,10*np.log10(255.0**2/np.mean((A-B)**2)),float(np.abs(d).max())))
    v=np.array([x[1] for x in r]); o=np.argsort(v)
    print(f"== {sc} n={len(v)} mean {v.mean():.3f} med {np.median(v):.3f}  min {v.min():.2f}  worst5: "+
          ", ".join(f"{r[i][0][:22]}={r[i][1]:.1f}dB(dc{r[i][2]:.0f})" for i in o[:5]))
    print(f"   mean if worst 10% frames were replaced by the median: {np.where(v<np.percentile(v,10),np.median(v),v).mean():.3f} (+{np.where(v<np.percentile(v,10),np.median(v),v).mean()-v.mean():.3f} dB)")

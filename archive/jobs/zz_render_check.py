import os, sys, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
dirs = {
 "r28m HCM0421": "/mnt/d/avv/r28_members/HCM0421/test_png",
 "r28m HCM0539": "/mnt/d/avv/r28_members/HCM0539/test_png",
 "r28m HCM0540": "/mnt/d/avv/r28_members/HCM0540/test_png",
 "r25mip HCM0539": "/mnt/d/avv/r25_mip3d/HCM0539/test_png",
 "r2r9 ut7 HCM0539": "/mnt/d/avv/r2r9/models/HCM0539_ut7/test_png",
}
for k,d in dirs.items():
    if not os.path.isdir(d):
        print(f"{k:>18}  MISSING {d}"); continue
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    nb=nw=tot=0; mn=[]; blackimg=0
    for f in fs:
        a = np.asarray(Image.open(os.path.join(d,f)).convert("RGB"))
        blk = (a.sum(2)==0)
        nb += int(blk.sum()); nw += int((a==255).all(2).sum()); tot += a.shape[0]*a.shape[1]
        mn.append(a.mean())
        if blk.mean()>0.5: blackimg+=1
    print(f"{k:>18}  n={len(fs)}  black px {nb/tot*100:.4f}%  sat-white {nw/tot*100:.4f}%  "
          f"mean {np.mean(mn):.3f}  per-img mean min/max {np.min(mn):.2f}/{np.max(mn):.2f}  blackimgs {blackimg}")

import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
from colmap_io import read_images_binary

for s in ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]:
    src=os.path.join(SET2,s); sp=find_sparse(src)
    imgs=read_images_binary(os.path.join(sp,"images.bin"))
    byname={v["name"]:v for v in imgs.values()}
    trn=sorted(os.listdir(os.path.join(src,"train/images")))
    te=load_poses_csv(os.path.join(src,"test/test_poses.csv")); ten=[t["name"] for t in te]
    extra=[n for n in byname if n not in set(trn) and n not in set(ten)]
    cnt={}
    for n in trn:
        for p in byname[n]["point3D_ids"]:
            if p>=0: cnt[int(p)]=cnt.get(int(p),0)+1
    # per test view: fraction of its points seen by 0 / 1 / <=2 train views
    f0,f1,f2,weak=[],[],[],[]
    for n in ten:
        pts=byname[n]["point3D_ids"]; pts=pts[pts>=0]
        c=np.array([cnt.get(int(p),0) for p in pts])
        f0.append((c==0).mean()); f1.append((c<=1).mean()); f2.append((c<=2).mean())
    f0=np.array(f0);f1=np.array(f1);f2=np.array(f2)
    nbad=(f0>0.05).sum()
    # do the extra (unused) poses sit near the under-supported test views?
    all_p,_=train_poses(src)
    Ce=centers([p for p in all_p if p["name"] in set(extra)]) if extra else None
    Ctr=centers([p for p in all_p if p["name"] in set(trn)])
    Cte=centers(te); spx=np.median(train_spacing(Ctr))
    msg=""
    if Ce is not None and len(Ce):
        de=np.linalg.norm(Cte[:,None,:]-Ce[None,:,:],axis=2).min(1)/spx
        bad=f0>0.05
        if bad.sum():
            msg=f" | dist-to-nearest-EXTRA-pose (train-spacings): under-supported views med={np.median(de[bad]):.2f} vs rest {np.median(de[~bad]):.2f}"
        else:
            msg=f" | extra-pose dist med={np.median(de):.2f}"
    print(f"{s:9s} n_extra_unused_poses={len(extra):3d} | test views with >5% unsupported pts: {nbad}/{len(ten)} "
          f"| frac pts seen by 0 train views: med={np.median(f0)*100:.2f}% p90={np.percentile(f0,90)*100:.2f}% max={f0.max()*100:.1f}%")
    print(f"          frac pts seen by <=1 train view: med={np.median(f1)*100:.1f}%  <=2: med={np.median(f2)*100:.1f}%{msg}")

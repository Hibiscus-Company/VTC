import os, sys, numpy as np
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
R='/mnt/d/avv/data/phase1/private_set2'
for sc,es in [('HCM0421','HCM0421'),('chair','chair'),('bonsai','bonsai'),('bonsai','bonsai2')]:
    sp=f'{R}/{sc}/train/sparse/0'
    imgs=read_extrinsics_binary(sp+'/images.bin')
    C={};D={}
    for im in imgs.values():
        Rm=qvec2rotmat(im.qvec); C[im.name]=-Rm.T@im.tvec; D[im.name]=Rm.T@np.array([0,0,1.0])
    tsub=sorted(os.listdir(f'/mnt/d/avv/evalsplit/{es}/train_sub/images'))
    ev=sorted(os.listdir(f'/mnt/d/avv/evalsplit/{es}/eval_gt'))
    tc=np.array([C[n] for n in tsub]); td=np.array([D[n] for n in tsub])
    nn=np.median(np.sort(np.linalg.norm(tc[:,None]-tc[None],axis=2),axis=1)[:,1])
    d=[];a=[]
    for n in ev:
        dd=np.linalg.norm(tc-C[n],axis=1); d.append(dd.min())
        a.append(np.degrees(np.arccos(np.clip(td@D[n],-1,1))).min())
    d=np.array(d)/nn; a=np.array(a)
    print(f"{es:9s} n_eval={len(ev)}  nearest-train-sub dist / NN-spacing: med {np.median(d):.2f} p90 {np.percentile(d,90):.2f} max {d.max():.2f} | frames >3x: {int((d>3).sum())} >6x: {int((d>6).sum())} | max view-angle {a.max():.1f}deg")

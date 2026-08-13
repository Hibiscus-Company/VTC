import sys, os, numpy as np, struct, cv2
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
def read_ids(p):
    ids=[]
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            b=f.read(43); ids.append(struct.unpack('<Q',b[:8])[0])
            tn=struct.unpack('<Q',f.read(8))[0]; f.read(8*tn)
    return np.array(ids)
R='/mnt/d/avv/data/phase1/private_set2'
for sc,osc in [('chair',1.5),('bonsai',1.0)]:
    sp=f'{R}/{sc}/train/sparse/0'; imd=f'{R}/{sc}/train/images'
    ids=read_ids(sp+'/points3D.bin'); pid2i={p:i for i,p in enumerate(ids)}
    imgs=read_extrinsics_binary(sp+'/images.bin'); onset=set(os.listdir(imd))
    N=len(ids); best=np.zeros(N,dtype=np.float32); cnt=np.zeros(N,dtype=np.int32); store=[]
    for im in sorted([i for i in imgs.values() if i.name in onset],key=lambda i:i.name):
        A=np.asarray(Image.open(os.path.join(imd,im.name)).convert('L'),dtype=np.float32)
        sm=cv2.boxFilter(np.abs(cv2.Laplacian(A,cv2.CV_32F)),-1,(33,33)); H,W=A.shape
        v=im.point3D_ids>=0; pid=im.point3D_ids[v]; xy=im.xys[v]/osc
        keep=np.array([p in pid2i for p in pid]); pid=pid[keep]; xy=xy[keep]
        x=np.clip(np.round(xy[:,0]).astype(int),0,W-1); y=np.clip(np.round(xy[:,1]).astype(int),0,H-1)
        idx=np.array([pid2i[p] for p in pid],dtype=np.int64); s=sm[y,x]
        np.maximum.at(best,idx,s); cnt[idx]+=1; store.append((idx,s,x,y,W,H))
    iqr=[];rng=[]
    for idx,s,x,y,W,H in store:
        m=cnt[idx]>=3
        rel=np.clip(s[m]/np.maximum(best[idx][m],1e-6),0,1)
        if len(rel)<50: continue
        iqr.append(np.percentile(rel,75)-np.percentile(rel,25))
        rng.append(np.percentile(rel,90)-np.percentile(rel,10))
    print(f"== {sc}: WITHIN-frame spread of content-controlled relative sharpness")
    print(f"   median per-frame IQR {np.median(iqr):.3f}, median p90-p10 {np.median(rng):.3f}   (0 => a per-FRAME scalar suffices; large => blur is spatial (DoF) and needs a per-PIXEL weight)")

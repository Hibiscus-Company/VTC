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
            b=f.read(43); pid=struct.unpack('<Q',b[:8])[0]
            tn=struct.unpack('<Q',f.read(8))[0]; f.read(8*tn); ids.append(pid)
    return np.array(ids)
R='/mnt/d/avv/data/phase1/private_set2'
for sc,osc in [('chair',1.5),('bonsai',1.0),('HCM0421',4.0)]:
    sp=f'{R}/{sc}/train/sparse/0'; imd=f'{R}/{sc}/train/images'
    ids=read_ids(sp+'/points3D.bin'); pid2i={p:i for i,p in enumerate(ids)}
    imgs=read_extrinsics_binary(sp+'/images.bin'); ondisk=sorted(os.listdir(imd))
    onset=set(ondisk)
    N=len(ids)
    S=np.zeros((N,), dtype=np.float32); best=np.zeros(N,dtype=np.float32); cnt=np.zeros(N,dtype=np.int32)
    obs_store=[]
    for im in sorted([i for i in imgs.values() if i.name in onset], key=lambda i:i.name):
        A=np.asarray(Image.open(os.path.join(imd,im.name)).convert('L'),dtype=np.float32)
        lap=np.abs(cv2.Laplacian(A,cv2.CV_32F))
        sm=cv2.boxFilter(lap,-1,(33,33))
        H,W=A.shape
        v=im.point3D_ids>=0
        pid=im.point3D_ids[v]; xy=im.xys[v]/osc
        keep=np.array([p in pid2i for p in pid])
        pid=pid[keep]; xy=xy[keep]
        x=np.clip(np.round(xy[:,0]).astype(int),0,W-1); y=np.clip(np.round(xy[:,1]).astype(int),0,H-1)
        s=sm[y,x]
        idx=np.array([pid2i[p] for p in pid],dtype=np.int64)
        np.maximum.at(best,idx,s); cnt[idx]+=1
        obs_store.append((im.name,idx,s))
    # per-frame content-controlled relative sharpness
    rows=[]
    for nm,idx,s in obs_store:
        m=cnt[idx]>=3
        rel=s[m]/np.maximum(best[idx][m],1e-6)
        rows.append((nm,float(np.median(rel))))
    rel=np.array([r[1] for r in rows])
    print(f"== {sc}: content-controlled per-frame relative sharpness (1.0 = sharpest view of its own content)")
    print(f"   med {np.median(rel):.3f}  p10 {np.percentile(rel,10):.3f}  p25 {np.percentile(rel,25):.3f}  p75 {np.percentile(rel,75):.3f}  p90 {np.percentile(rel,90):.3f}")
    print(f"   frames with rel<0.5 (their content is >2x sharper somewhere else): {np.mean(rel<0.5):.3f}  rel<0.35: {np.mean(rel<0.35):.3f}")
    # coverage: per point, how many observations are within 80% of its best sharpness
    good=[]
    Nsharp=np.zeros(N,dtype=np.int32)
    for nm,idx,s in obs_store:
        Nsharp[idx]+= (s>=0.8*best[idx]).astype(np.int32)
    m=cnt>=3
    print(f"   per-point coverage by NEAR-SHARPEST views (>=80% of its best): mean {Nsharp[m].mean():.2f} views, "
          f"frac points with >=2 such views {np.mean(Nsharp[m]>=2):.3f}, >=3: {np.mean(Nsharp[m]>=3):.3f}")
    np.save(f'/tmp/relsharp_{sc}.npy', np.array([[r[1]] for r in rows]))
    with open(f'/tmp/relsharp_{sc}.txt','w') as f:
        for nm,v in rows: f.write(f"{nm}\t{v:.4f}\n")

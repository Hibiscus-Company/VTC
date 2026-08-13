import os, sys, csv, numpy as np
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
R='/mnt/d/avv/data/phase1/private_set2'
def q2R(q):
    return qvec2rotmat(np.array(q,dtype=float))
for sc in sorted(os.listdir(R)):
    sp=f'{R}/{sc}/train/sparse/0'
    imgs=read_extrinsics_binary(sp+'/images.bin')
    ondisk=set(os.listdir(f'{R}/{sc}/train/images'))
    trC=[];trD=[]
    for im in imgs.values():
        if im.name in ondisk:
            Rm=qvec2rotmat(im.qvec); trC.append(-Rm.T@im.tvec); trD.append(Rm.T@np.array([0,0,1.0]))
    trC=np.array(trC); trD=np.array(trD)
    ctr=trC.mean(0); rad=np.linalg.norm(trC-ctr,axis=1)
    nn=np.sort(np.linalg.norm(trC[:,None]-trC[None],axis=2),axis=1)[:,1]
    # test poses
    cp=f'{R}/{sc}/test/test_poses.csv'
    rows=list(csv.DictReader(open(cp)))
    k=list(rows[0].keys())
    def get(r,names):
        for n in names:
            for kk in k:
                if kk.strip().lower()==n: return float(r[kk])
        return None
    out=[]
    for r in rows:
        q=[get(r,[x]) for x in ['qw','qx','qy','qz']]; t=[get(r,[x]) for x in ['tx','ty','tz']]
        if None in q or None in t: out=None; break
        Rm=q2R(q); C=-Rm.T@np.array(t); D=Rm.T@np.array([0,0,1.0])
        d=np.linalg.norm(trC-C,axis=1); ang=np.degrees(np.arccos(np.clip(trD@D,-1,1)))
        out.append((np.linalg.norm(C-ctr), d.min(), ang.min()))
    if out is None: print(sc,'csv keys',k); continue
    o=np.array(out)
    ext=o[:,1]/np.median(nn)
    print(f"{sc:9s} n_test={len(o):3d} | train radius med {np.median(rad):.2f} max {rad.max():.2f} | test dist-to-centre med {np.median(o[:,0]):.2f} max {o[:,0].max():.2f} "
          f"| nearest-train-cam dist  med {np.median(o[:,1]):.3f} p95 {np.percentile(o[:,1],95):.3f} max {o[:,1].max():.3f}  (train NN spacing med {np.median(nn):.3f})")
    print(f"{'':9s} EXTRAPOLATION: test poses with nearest-train dist > 3x train NN spacing: {int((ext>3).sum())}/{len(o)}  >6x: {int((ext>6).sum())} | max view-angle-to-nearest {o[:,2].max():.1f} deg, n>15deg: {int((o[:,2]>15).sum())}")

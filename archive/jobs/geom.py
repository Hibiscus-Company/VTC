import numpy as np, csv, os, re, struct, sys
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
D='/mnt/d/avv/data/phase1/private_set2'
def qvec2R(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
def read_images_bin(p):
    out={}
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            iid,qw,qx,qy,qz,tx,ty,tz,cid=struct.unpack('<idddddddi',f.read(64))
            name=b''
            while True:
                c=f.read(1)
                if c==b'\x00': break
                name+=c
            npts=struct.unpack('<Q',f.read(8))[0]
            f.read(24*npts)
            out[name.decode()]=(np.array([qw,qx,qy,qz]),np.array([tx,ty,tz]))
    return out
for scene in ['HCM0421','HCM0539','HCM0540','HCM0644','HCM0674','chair','bonsai']:
    ib=os.path.join(D,scene,'train','sparse','0','images.bin')
    if not os.path.exists(ib): print(scene,"NO images.bin"); continue
    tr=read_images_bin(ib)
    trn=sorted(tr.keys())
    C_tr=np.array([-qvec2R(tr[n][0]).T@tr[n][1] for n in trn])
    Z_tr=np.array([qvec2R(tr[n][0]).T@np.array([0,0,1.0]) for n in trn])
    rows=list(csv.DictReader(open(os.path.join(D,scene,'test','test_poses.csv'))))
    C_te=np.array([-qvec2R([float(r['qw']),float(r['qx']),float(r['qy']),float(r['qz'])]).T@
                   np.array([float(r['tx']),float(r['ty']),float(r['tz'])]) for r in rows])
    Z_te=np.array([qvec2R([float(r['qw']),float(r['qx']),float(r['qy']),float(r['qz'])]).T@np.array([0,0,1.0]) for r in rows])
    # scene scale = median dist of train cams to centroid
    scale=np.median(np.linalg.norm(C_tr-C_tr.mean(0),axis=1))
    d=np.linalg.norm(C_te[:,None,:]-C_tr[None,:,:],axis=2)   # 58x240
    nn=np.sort(d,axis=1)[:,:3]
    ang=np.degrees(np.arccos(np.clip(np.einsum('id,jd->ij',Z_te,Z_tr),-1,1)))
    nnidx=np.argmin(d,axis=1)
    angnn=ang[np.arange(len(C_te)),nnidx]
    # train-train nearest spacing
    dtt=np.linalg.norm(C_tr[:,None,:]-C_tr[None,:,:],axis=2); np.fill_diagonal(dtt,1e9)
    tt=dtt.min(1)
    print(f"\n=== {scene}  (scale={scale:.3f})")
    print(f"  train nn-spacing   median {np.median(tt)/scale*100:5.2f}% of scale")
    print(f"  test->train nn1    median {np.median(nn[:,0])/scale*100:5.2f}%   p90 {np.percentile(nn[:,0],90)/scale*100:5.2f}%   max {nn[:,0].max()/scale*100:5.2f}%")
    print(f"  test->train nn3    median {np.median(nn[:,2])/scale*100:5.2f}%")
    print(f"  ratio nn1/train-spacing  median {np.median(nn[:,0])/np.median(tt):.2f}")
    print(f"  view-dir angle to nn      median {np.median(angnn):5.2f} deg   p90 {np.percentile(angnn,90):5.2f}  max {angnn.max():5.2f}")

import sys, os, numpy as np, struct
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
def read_pts(p):
    xyz=[];ids=[]
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            pid,x,y,z,r,g,b,e=struct.unpack('<QdddBBBd',f.read(43))
            tn=struct.unpack('<Q',f.read(8))[0]; f.read(8*tn)
            ids.append(pid);xyz.append((x,y,z))
    return np.array(ids),np.array(xyz)
R='/mnt/d/avv/data/phase1/private_set2'
for sc,obs_scale in [('HCM0421',4.0),('bonsai',1.0),('chair',1.5)]:
    sp=f'{R}/{sc}/train/sparse/0'
    ids,xyz=read_pts(sp+'/points3D.bin'); pid2i={p:i for i,p in enumerate(ids)}
    cams=read_intrinsics_binary(sp+'/cameras.bin'); c=list(cams.values())[0]
    par=np.array(c.params,dtype=float)
    if c.model in ('SIMPLE_RADIAL','SIMPLE_PINHOLE'):
        fx=fy=par[0]; cx,cy=par[1],par[2]; k1=par[3] if c.model=='SIMPLE_RADIAL' else 0.0
    imgs=read_extrinsics_binary(sp+'/images.bin'); ondisk=set(os.listdir(f'{R}/{sc}/train/images'))
    W,H=c.width,c.height
    allr=[];perimg=[]
    order=sorted([im for im in imgs.values() if im.name in ondisk], key=lambda i:i.name)
    Cs=[];Rs=[]
    for im in order:
        Rm=qvec2rotmat(im.qvec); Cs.append(-Rm.T@im.tvec); Rs.append(Rm)
        v=im.point3D_ids>=0
        pids=im.point3D_ids[v]; obs=im.xys[v]/obs_scale
        keep=np.array([p in pid2i for p in pids]); pids=pids[keep]; obs=obs[keep]
        P=xyz[[pid2i[p] for p in pids]]
        X=(Rm@P.T).T+im.tvec
        z=X[:,2]; m=z>1e-6
        X=X[m]; obs=obs[m]; z=z[m]
        u=X[:,0]/z; v2=X[:,1]/z; r2=u*u+v2*v2
        d=1+k1*r2
        px=fx*u*d+cx; py=fy*v2*d+cy
        res=np.c_[obs[:,0]-px, obs[:,1]-py]
        good=np.linalg.norm(res,axis=1)<5
        allr.append(np.c_[px[good],py[good],res[good]])
        perimg.append((im.name, px[good],py[good],res[good]))
    A=np.concatenate(allr)
    print(f"== {sc} ({c.model}) n_obs={len(A)}  median |res| {np.median(np.linalg.norm(A[:,2:],axis=1)):.3f}px  mean res {A[:,2:].mean(0).round(4)}")
    # radial profile of the MEAN residual (unmodelled distortion signature)
    rr=np.hypot(A[:,0]-cx,A[:,1]-cy); rmax=np.hypot(cx,cy)
    ur=np.c_[(A[:,0]-cx)/np.maximum(rr,1e-6),(A[:,1]-cy)/np.maximum(rr,1e-6)]
    radial=(A[:,2]*ur[:,0]+A[:,3]*ur[:,1])
    bins=np.linspace(0,rmax,9); bi=np.digitize(rr,bins)-1
    prof=[np.mean(radial[bi==i]) if (bi==i).sum()>50 else np.nan for i in range(8)]
    print(f"   MEAN RADIAL residual by r/rmax bin (0..1): {np.round(prof,3)}  px  [+ = obs further out than model]")
    tang=(-A[:,2]*ur[:,1]+A[:,3]*ur[:,0])
    proft=[np.mean(tang[bi==i]) if (bi==i).sum()>50 else np.nan for i in range(8)]
    print(f"   MEAN TANGENTIAL residual by bin:          {np.round(proft,3)}  px")
    # rolling shutter: per-image slope of dx,dy vs row, vs angular velocity
    Cs=np.array(Cs); Rs=np.array(Rs)
    om=[]
    for i in range(len(Rs)):
        j=min(i+1,len(Rs)-1); i0=max(i-1,0)
        dR=Rs[j]@Rs[i0].T
        ang=np.arccos(np.clip((np.trace(dR)-1)/2,-1,1))/max(j-i0,1)
        om.append(np.degrees(ang))
    om=np.array(om)
    sl=[]
    for (nm,px,py,res) in perimg:
        Xd=np.c_[np.ones(len(py)),(py-cy)/H]
        bx=np.linalg.lstsq(Xd,res[:,0],rcond=None)[0][1]
        by=np.linalg.lstsq(Xd,res[:,1],rcond=None)[0][1]
        sl.append((bx,by))
    sl=np.array(sl)
    mag=np.hypot(sl[:,0],sl[:,1])
    print(f"   ROLLING SHUTTER probe: |row-linear residual slope| med {np.median(mag):.3f}px/frame-height  p90 {np.percentile(mag,90):.3f}")
    print(f"     corr(|slope|, |angular vel| per frame) = {np.corrcoef(mag,om)[0,1]:+.3f}   (omega med {np.median(om):.3f} deg/frame)")

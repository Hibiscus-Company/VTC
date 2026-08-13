import sys, os, numpy as np, struct
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from PIL import Image
Image.MAX_IMAGE_PIXELS=None

def read_pts(p):
    xyz=[];rgb=[];err=[];tl=[];ids=[]
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            pid,x,y,z,r,g,b,e=struct.unpack('<QdddBBBd',f.read(43))
            tn=struct.unpack('<Q',f.read(8))[0]
            f.read(8*tn)
            ids.append(pid);xyz.append((x,y,z));rgb.append((r,g,b));err.append(e);tl.append(tn)
    return np.array(ids),np.array(xyz),np.array(rgb),np.array(err),np.array(tl)

R='/mnt/d/avv/data/phase1/private_set2'
for sc,obs_scale in [('HCM0421',4.0),('bonsai',1.0),('chair',1.5)]:
    sp=f'{R}/{sc}/train/sparse/0'; imd=f'{R}/{sc}/train/images'
    ids,xyz,rgb,err,tl=read_pts(sp+'/points3D.bin')
    imgs=read_extrinsics_binary(sp+'/images.bin'); ondisk=set(os.listdir(imd))
    ntr=sum(1 for im in imgs.values() if im.name in ondisk)
    print(f"== {sc}: {len(ids)} pts, track len med {np.median(tl):.0f} p10 {np.percentile(tl,10):.0f} p90 {np.percentile(tl,90):.0f} "
          f"| frac tl<=3: {np.mean(tl<=3):.3f} | reproj err med {np.median(err):.3f}px p90 {np.percentile(err,90):.3f}")
    pid2i={p:i for i,p in enumerate(ids)}
    # per-track TRAIN-ONLY observation count
    cnt=np.zeros(len(ids),dtype=np.int32)
    names=[]; obsl=[]
    for im in imgs.values():
        if im.name not in ondisk: continue
        names.append(im.name)
        v=im.point3D_ids>=0
        idx=np.array([pid2i[p] for p in im.point3D_ids[v] if p in pid2i],dtype=np.int64)
        cnt[idx]+=1
        obsl.append((im.name, im.xys[v]/obs_scale, idx))
    print(f"   train-only track len: med {np.median(cnt[cnt>0]):.0f}, frac with <=2 train obs {np.mean(cnt[cnt>0]<=2):.3f}, "
          f"pts unseen by any train img {np.mean(cnt==0):.4f}")
    # photometric consistency across the track
    Ccol=np.zeros((len(ids),3)); Cn=np.zeros(len(ids))
    per=[]
    for nm,xy,idx in obsl:
        A=np.asarray(Image.open(os.path.join(imd,nm)).convert('RGB'),dtype=np.float32)
        H,W,_=A.shape
        x=np.clip(np.round(xy[:,0]).astype(int),0,W-1); y=np.clip(np.round(xy[:,1]).astype(int),0,H-1)
        c=A[y,x]
        per.append((nm,idx,c))
        np.add.at(Ccol,idx,c); np.add.at(Cn,idx,1)
    mean=Ccol/np.maximum(Cn,1)[:,None]
    good=Cn>=4
    resid=[];gains=[]
    for nm,idx,c in per:
        m=good[idx]
        d=c[m]-mean[idx][m]
        lum_c=c[m].mean(1); lum_m=mean[idx][m].mean(1)
        A_=np.c_[lum_m,np.ones(len(lum_m))]
        g,b=np.linalg.lstsq(A_,lum_c,rcond=None)[0]
        gains.append((nm,g,b,np.abs(d).mean()))
        resid.append((np.abs(d).mean(), np.abs(lum_c-(g*lum_m+b)).mean()))
    r=np.array(resid); G=np.array([[x[1],x[2]] for x in gains])
    print(f"   track colour MAD: raw {r[:,0].mean():.2f} levels -> after per-frame gain+bias {r[:,1].mean():.2f} levels "
          f"({100*(1-r[:,1].mean()/r[:,0].mean()):.0f}% explained by exposure)")
    print(f"   per-frame gain spread std {G[:,0].std():.4f} (range {G[:,0].min():.3f}-{G[:,0].max():.3f}), bias std {G[:,1].std():.2f} levels")

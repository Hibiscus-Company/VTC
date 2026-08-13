"""Three free measurements, each kills or saves a different branch.
 M1 aspect-ratio histogram  -> was my "sub-pixel" verdict an artifact of taking sigma_MINOR?
 M2 VoL single vs ensemble  -> is the blur evidence confounded by ensembling?
 M3 beta_t blur score + direction -> does group 4 (per-frame blur forward model) live or die?
"""
import os,sys,struct,numpy as np,torch,cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None; torch.set_num_threads(4)
exec(open("freqmatch.py").read().split("def tex_period")[0].split("import os,")[1].join(["import os,",""])) if False else None
def qvec2R(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]],dtype=np.float64)
def read_images_bin(p):
    out={}
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            d=struct.unpack('<idddddddi',f.read(64)); nm=b''
            while True:
                c=f.read(1)
                if c==b'\x00':break
                nm+=c
            k=struct.unpack('<Q',f.read(8))[0]; f.read(24*k)
            out[nm.decode()]=(np.array(d[1:5]),np.array(d[5:8]))
    return out

# ---------------- M1: aspect ratio, 3D and projected 2D ----------------
def m1(tag,ckpt,scene,cam,nv=2):
    c=torch.load(ckpt,map_location="cpu",weights_only=False); sp=c["splats"]
    sc=torch.exp(sp["scales"]); op=torch.sigmoid(sp["opacities"]); keep=op>0.05
    S=sc[keep].double().numpy(); mu=sp["means"][keep].double().numpy(); Q=sp["quats"][keep].double().numpy()
    ar3=S.max(1)/np.maximum(S.min(1),1e-12)
    print(f"\n[M1] {tag}  N(op>0.05)={len(S):,}")
    print(f"   3D scale aspect (max/min axis): med {np.median(ar3):.2f}  p75 {np.percentile(ar3,75):.2f}  "
          f"p95 {np.percentile(ar3,95):.2f}  frac>4 {100*np.mean(ar3>4):.1f}%  frac>10 {100*np.mean(ar3>10):.1f}%")
    D=f"/mnt/d/avv/data/phase1/private_set2/{scene}"
    imgs=read_images_bin(f"{D}/train/sparse/0/images.bin"); have=set(os.listdir(f"{D}/train/images"))
    names=sorted(n for n in imgs if n in have)[::max(1,len(have)//nv)][:nv]
    fx,fy,cx,cy=cam
    AR=[];MAJ=[];MIN=[]
    for nm in names:
        q,t=imgs[nm]; Rw=qvec2R(q)
        Xc=mu@Rw.T+t; z=Xc[:,2]; m=z>0.2
        if m.sum()>800000:
            idx=np.random.RandomState(0).choice(np.where(m)[0],800000,replace=False)
        else: idx=np.where(m)[0]
        Xc=Xc[idx];Ss=S[idx];Qq=Q[idx];zz=Xc[:,2]
        u=fx*Xc[:,0]/zz+cx; v=fy*Xc[:,1]/zz+cy
        inb=(u>0)&(u<cx*2)&(v>0)&(v<cy*2)
        Xc=Xc[inb];Ss=Ss[inb];Qq=Qq[inb];zz=zz[inb]
        Qn=Qq/np.linalg.norm(Qq,axis=1,keepdims=True); w_,x_,y_,z_=Qn.T
        Rg=np.empty((len(Qn),3,3))
        Rg[:,0,0]=1-2*(y_*y_+z_*z_);Rg[:,0,1]=2*(x_*y_-w_*z_);Rg[:,0,2]=2*(x_*z_+w_*y_)
        Rg[:,1,0]=2*(x_*y_+w_*z_); Rg[:,1,1]=1-2*(x_*x_+z_*z_);Rg[:,1,2]=2*(y_*z_-w_*x_)
        Rg[:,2,0]=2*(x_*z_-w_*y_); Rg[:,2,1]=2*(y_*z_+w_*x_); Rg[:,2,2]=1-2*(x_*x_+y_*y_)
        M=Rg*Ss[:,None,:]; Sig=M@np.transpose(M,(0,2,1)); Sc=Rw@Sig@Rw.T
        J=np.zeros((len(zz),2,3)); J[:,0,0]=fx/zz;J[:,0,2]=-fx*Xc[:,0]/zz**2
        J[:,1,1]=fy/zz;J[:,1,2]=-fy*Xc[:,1]/zz**2
        S2=J@Sc@np.transpose(J,(0,2,1))
        a=S2[:,0,0]+0.3;b=S2[:,0,1];d=S2[:,1,1]+0.3
        tr=a+d;det=a*d-b*b;disc=np.sqrt(np.maximum(tr*tr/4-det,0))
        lmaj=np.maximum(tr/2+disc,1e-8);lmin=np.maximum(tr/2-disc,1e-8)
        AR.append(np.sqrt(lmaj/lmin));MAJ.append(np.sqrt(lmaj));MIN.append(np.sqrt(lmin))
    AR=np.concatenate(AR);MAJ=np.concatenate(MAJ);MIN=np.concatenate(MIN)
    print(f"   PROJECTED 2D: sigma_minor med {np.median(MIN):.3f}px   sigma_MAJOR med {np.median(MAJ):.3f}px"
          f"   aspect med {np.median(AR):.2f} p90 {np.percentile(AR,90):.2f}")
    print(f"   sigma_MAJOR: frac>1px {100*np.mean(MAJ>1):.1f}%  frac>2px {100*np.mean(MAJ>2):.1f}%  "
          f"frac>4px {100*np.mean(MAJ>4):.1f}%")
    return MAJ,MIN,AR

# ---------------- M2: VoL single member vs ensemble mean vs GT ----------------
def vol(a): return float(cv2.Laplacian(a.astype(np.float32),cv2.CV_32F).var())
def m2():
    print("\n[M2] bonsai VoL  (Laplacian variance; higher = sharper)")
    MEM=["/mnt/d/avv/r14/bonsai_aa42/test_png","/mnt/d/avv/r14/bonsai_aa7/test_png",
         "/mnt/d/avv/r14/bonsai_aa13/test_png","/mnt/d/avv/r24_bonsai/aa101/test_png",
         "/mnt/d/avv/r24_bonsai/aa202/test_png","/mnt/d/avv/r24_bonsai/aa303/test_png",
         "/mnt/d/avv/r28_members/bonsai/test_png"]
    MEM=[m for m in MEM if os.path.isdir(m)]
    st=sorted(f[:-4] for f in os.listdir(MEM[0]) if f.endswith(".png"))[:8]
    g=lambda p:np.asarray(Image.open(p).convert("L"),dtype=np.float32)
    vs,ve=[],[]
    for s in st:
        ims=[g(f"{m}/{s}.png") for m in MEM]
        vs.append(np.mean([vol(i) for i in ims])); ve.append(vol(np.mean(ims,0)))
    print(f"   single member (mean over {len(MEM)}): {np.mean(vs):7.2f}")
    print(f"   ensemble mean               : {np.mean(ve):7.2f}   ratio ens/single {np.mean(ve)/np.mean(vs):.3f}")
    D="/mnt/d/avv/data/phase1/private_set2/bonsai"
    tr="/mnt/d/avv/blurbound/bonsai/train_png"
    if os.path.isdir(tr):
        stt=sorted(f[:-4] for f in os.listdir(tr) if f.endswith(".png"))[:12]
        vr=[vol(g(f"{tr}/{s}.png")) for s in stt]
        vg=[vol(g(f"{D}/train/images/{s}.jpg")) for s in stt if os.path.exists(f"{D}/train/images/{s}.jpg")]
        print(f"   TRAIN single render {np.mean(vr):7.2f}   TRAIN GT photo {np.mean(vg):7.2f}   ratio {np.mean(vr)/np.mean(vg):.3f}")

# ---------------- M3: beta_t blur score + blur direction ----------------
def m3(scene,ext,cap=None):
    D=f"/mnt/d/avv/data/phase1/private_set2/{scene}/train/images"
    fs=sorted(os.listdir(D)); fs=fs[:cap] if cap else fs
    bs=[];ang=[];aniso=[]
    for f in fs:
        a=np.asarray(Image.open(f"{D}/{f}").convert("L"),dtype=np.float32)/255.
        a=cv2.resize(a,(a.shape[1]//2,a.shape[0]//2),interpolation=cv2.INTER_AREA)
        F=np.abs(np.fft.fftshift(np.fft.fft2(a-a.mean())))**2
        H,W=F.shape;cy,cx=H//2,W//2
        Y,X=np.ogrid[:H,:W];R=np.sqrt(((Y-cy)/cy)**2+((X-cx)/cx)**2)
        hi=(R>0.35)&(R<=1.0)
        bs.append(float(F[hi].sum()/max(F.sum(),1e-12)))
        gx=cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)
        Jxx=float((gx*gx).mean());Jyy=float((gy*gy).mean());Jxy=float((gx*gy).mean())
        tr=Jxx+Jyy;dd=np.sqrt(max((Jxx-Jyy)**2+4*Jxy**2,0))
        l1=(tr+dd)/2;l2=(tr-dd)/2
        aniso.append(float((l1-l2)/max(l1+l2,1e-12)))
        ang.append(float(np.degrees(0.5*np.arctan2(2*Jxy,Jxx-Jyy))%180))
    bs=np.array(bs);ang=np.array(ang);aniso=np.array(aniso)
    print(f"\n[M3] {scene}  n={len(fs)} train frames")
    print(f"   beta_t (HF energy frac): med {np.median(bs):.5f}  CV {bs.std()/bs.mean():.3f}  "
          f"p5 {np.percentile(bs,5):.5f}  p95 {np.percentile(bs,95):.5f}  ratio p95/p5 {np.percentile(bs,95)/max(np.percentile(bs,5),1e-9):.2f}")
    print(f"   gradient anisotropy    : med {np.median(aniso):.4f}  (0=isotropic, 1=fully directional)")
    print(f"   dominant orientation   : circ-spread {ang.std():.1f} deg  (large=varies per frame)")
    return bs,ang,aniso

CAM_B=(1108.5124,1108.5124,960.0,540.0)
import csv
r=next(csv.DictReader(open("/mnt/d/avv/data/phase1/private_set2/HCM0421/test/test_poses.csv")))
CAM_T=(float(r["fx"]),float(r["fy"]),float(r["cx"]),float(r["cy"]))
m1("BONSAI capD 5M","/mnt/d/avv/bonsai_sel/capD_5Mearly/ckpt.pt","bonsai",CAM_B)
m1("TOWER HCM0421 ut42","/mnt/d/avv/r2r9/models/HCM0421_ut42/ckpt.pt","HCM0421",CAM_T)
m2()
m3("bonsai",".jpg")
m3("chair",".jpg")

"""D1 disc-or-needle (3D, projection-free) | D2 beta_t vs camera velocity | D3 axis-vs-blur alignment"""
import os,struct,numpy as np,torch,cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None; torch.set_num_threads(4)
def qvec2R(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]],dtype=np.float64)
def rib(p):
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

print("="*96); print("D1  DISC OR NEEDLE  (3D scales, sorted s1>=s2>=s3, projection-free)")
for tag,ck in [("bonsai capD 5M","/mnt/d/avv/bonsai_sel/capD_5Mearly/ckpt.pt"),
               ("tower ut42 8M","/mnt/d/avv/r2r9/models/HCM0421_ut42/ckpt.pt")]:
    c=torch.load(ck,map_location="cpu",weights_only=False); sp=c["splats"]
    S=torch.exp(sp["scales"]); op=torch.sigmoid(sp["opacities"])
    S=S[op>0.05].double().numpy()
    Ss=np.sort(S,axis=1)[:,::-1]           # s1>=s2>=s3
    r21=Ss[:,1]/np.maximum(Ss[:,0],1e-30)  # ->1 means DISC (s2~s1)
    r32=Ss[:,1]/np.maximum(Ss[:,2],1e-30)  # ->1 means NEEDLE (s2~s3)
    q=lambda a,p:np.percentile(a,p)
    print(f"\n {tag}  N={len(S):,}")
    print(f"   s2/s1  (->1 = DISC)   med {np.median(r21):.4f}   p10 {q(r21,10):.4f}  p25 {q(r21,25):.4f}  p75 {q(r21,75):.4f}  p90 {q(r21,90):.4f}")
    print(f"   s2/s3  (->1 = NEEDLE) med {np.median(r32):.2f}   p10 {q(r32,10):.2f}  p25 {q(r32,25):.2f}  p75 {q(r32,75):.2f}  p90 {q(r32,90):.2f}")
    disc=np.mean(r21>0.5); need=np.mean(r32<2.0)
    print(f"   classify: DISC-like (s2/s1>0.5) {100*disc:.1f}%   NEEDLE-like (s2/s3<2) {100*need:.1f}%   neither {100*(1-disc-need):.1f}%")

D="/mnt/d/avv/data/phase1/private_set2"
def betas(scene,cap=None):
    P=f"{D}/{scene}/train/images"; fs=sorted(os.listdir(P)); fs=fs[:cap] if cap else fs
    b=[];ang=[]
    for f in fs:
        a=np.asarray(Image.open(f"{P}/{f}").convert("L"),dtype=np.float32)/255.
        a=cv2.resize(a,(a.shape[1]//2,a.shape[0]//2),interpolation=cv2.INTER_AREA)
        F=np.abs(np.fft.fftshift(np.fft.fft2(a-a.mean())))**2
        H,W=F.shape;cy,cx=H//2,W//2;Y,X=np.ogrid[:H,:W]
        R=np.sqrt(((Y-cy)/cy)**2+((X-cx)/cx)**2)
        b.append(float(F[(R>0.35)&(R<=1.0)].sum()/max(F.sum(),1e-12)))
        gx=cv2.Sobel(a,cv2.CV_32F,1,0,3);gy=cv2.Sobel(a,cv2.CV_32F,0,1,3)
        Jxx=float((gx*gx).mean());Jyy=float((gy*gy).mean());Jxy=float((gx*gy).mean())
        ang.append(float(np.degrees(0.5*np.arctan2(2*Jxy,Jxx-Jyy))%180))
    return fs,np.array(b),np.array(ang)

print("\n"+"="*96); print("D2  IS beta_t BLUR OR CONTENT?  (correlate with camera velocity from COLMAP)")
for scene in ["bonsai","chair"]:
    fs,b,ang=betas(scene)
    im=rib(f"{D}/{scene}/train/sparse/0/images.bin")
    C=[];Z=[]
    for f in fs:
        q_,t_=im[f]; R=qvec2R(q_); C.append(-R.T@t_); Z.append(R.T@np.array([0,0,1.0]))
    C=np.array(C);Z=np.array(Z)
    v=np.zeros(len(C)); w=np.zeros(len(C))
    v[1:-1]=np.linalg.norm(C[2:]-C[:-2],axis=1)/2
    w[1:-1]=np.degrees(np.arccos(np.clip(np.einsum('id,id->i',Z[2:],Z[:-2]),-1,1)))/2
    v[0]=v[1];v[-1]=v[-2];w[0]=w[1];w[-1]=w[-2]
    m=np.isfinite(v)&np.isfinite(b)
    cv_=np.corrcoef(v[m],b[m])[0,1]; cw=np.corrcoef(w[m],b[m])[0,1]
    sp=lambda a,c:np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(c)))[0,1]
    print(f"\n {scene}: n={len(fs)}  beta_t med {np.median(b):.5f}  p95/p5 {np.percentile(b,95)/np.percentile(b,5):.2f}x")
    print(f"   corr(beta_t, translational speed) r={cv_:+.3f}  spearman {sp(v[m],b[m]):+.3f}")
    print(f"   corr(beta_t, angular speed)       r={cw:+.3f}  spearman {sp(w[m],b[m]):+.3f}")
    print(f"   READ: strongly NEGATIVE r => faster camera -> less HF -> beta_t IS motion blur.")
    print(f"         near-zero r        => beta_t is CONTENT, group 4 ungated.")

print("\n"+"="*96); print("D3  DOES THE GAUSSIAN MAJOR AXIS ALIGN WITH THE BLUR DIRECTION? (bonsai)")
c=torch.load("/mnt/d/avv/bonsai_sel/capD_5Mearly/ckpt.pt",map_location="cpu",weights_only=False)
sp=c["splats"]; op=torch.sigmoid(sp["opacities"]); keep=op>0.05
mu=sp["means"][keep].double().numpy(); S=torch.exp(sp["scales"])[keep].double().numpy(); Q=sp["quats"][keep].double().numpy()
im=rib(f"{D}/bonsai/train/sparse/0/images.bin"); have=set(os.listdir(f"{D}/bonsai/train/images"))
fs,b,ang=betas("bonsai",cap=None)
names=[f for f in fs if f in have][::20][:10]
fx=fy=1108.5124;cx,cy=960.0,540.0
rng=np.random.RandomState(0); sub=rng.choice(len(mu),min(300000,len(mu)),replace=False)
mu_,S_,Q_=mu[sub],S[sub],Q[sub]
diffs=[]
for nm in names:
    q_,t_=im[nm]; Rw=qvec2R(q_)
    Xc=mu_@Rw.T+t_; z=Xc[:,2]; m=z>0.2
    Xc=Xc[m];Ss=S_[m];Qq=Q_[m];zz=Xc[:,2]
    u=fx*Xc[:,0]/zz+cx; v=fy*Xc[:,1]/zz+cy
    ib=(u>0)&(u<1920)&(v>0)&(v<1080)
    Xc=Xc[ib];Ss=Ss[ib];Qq=Qq[ib];zz=zz[ib]
    if len(Xc)<1000: continue
    Qn=Qq/np.linalg.norm(Qq,axis=1,keepdims=True);w_,x_,y_,z_=Qn.T
    Rg=np.empty((len(Qn),3,3))
    Rg[:,0,0]=1-2*(y_*y_+z_*z_);Rg[:,0,1]=2*(x_*y_-w_*z_);Rg[:,0,2]=2*(x_*z_+w_*y_)
    Rg[:,1,0]=2*(x_*y_+w_*z_); Rg[:,1,1]=1-2*(x_*x_+z_*z_);Rg[:,1,2]=2*(y_*z_-w_*x_)
    Rg[:,2,0]=2*(x_*z_-w_*y_); Rg[:,2,1]=2*(y_*z_+w_*x_); Rg[:,2,2]=1-2*(x_*x_+y_*y_)
    M=Rg*Ss[:,None,:];Sig=M@np.transpose(M,(0,2,1));Sc=Rw@Sig@Rw.T
    J=np.zeros((len(zz),2,3));J[:,0,0]=fx/zz;J[:,0,2]=-fx*Xc[:,0]/zz**2
    J[:,1,1]=fy/zz;J[:,1,2]=-fy*Xc[:,1]/zz**2
    S2=J@Sc@np.transpose(J,(0,2,1))
    a=S2[:,0,0]+0.3;bb=S2[:,0,1];d=S2[:,1,1]+0.3
    th=np.degrees(0.5*np.arctan2(2*bb,a-d))%180          # gaussian major-axis orientation
    i=fs.index(nm); blur=(ang[i]+90.0)%180               # blur dir = perp to max-gradient dir
    dd=np.abs(((th-blur+90)%180)-90)                      # circular diff in [0,90]
    diffs.append(np.median(dd))
diffs=np.array(diffs)
print(f"   n={len(diffs)} frames; median |angle(gaussian major) - angle(blur)| = {np.median(diffs):.1f} deg")
print(f"   45 deg = NO alignment (random).  <30 = aligned (blur absorbed into gaussian shape).")

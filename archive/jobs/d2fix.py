"""D2 REDONE: velocity normalised by frame-index gap (gaps are 10/20/30 bonsai, 5/10/15/20 chair)."""
import os,re,struct,numpy as np,cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
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
            k=struct.unpack('<Q',f.read(8))[0];f.read(24*k)
            out[nm.decode()]=(np.array(d[1:5]),np.array(d[5:8]))
    return out
D="/mnt/d/avv/data/phase1/private_set2"
for scene in ["bonsai","chair"]:
    P=f"{D}/{scene}/train/images"; fs=sorted(os.listdir(P))
    idx=np.array([int(re.findall(r'(\d+)',f)[0]) for f in fs],dtype=float)
    im=rib(f"{D}/{scene}/train/sparse/0/images.bin")
    C=[];Z=[]
    for f in fs:
        q_,t_=im[f];R=qvec2R(q_);C.append(-R.T@t_);Z.append(R.T@np.array([0,0,1.0]))
    C=np.array(C);Z=np.array(Z)
    b=[]
    for f in fs:
        a=np.asarray(Image.open(f"{P}/{f}").convert("L"),dtype=np.float32)/255.
        a=cv2.resize(a,(a.shape[1]//2,a.shape[0]//2),interpolation=cv2.INTER_AREA)
        F=np.abs(np.fft.fftshift(np.fft.fft2(a-a.mean())))**2
        H,W=F.shape;cy,cx=H//2,W//2;Y,X=np.ogrid[:H,:W]
        R_=np.sqrt(((Y-cy)/cy)**2+((X-cx)/cx)**2)
        b.append(float(F[(R_>0.35)&(R_<=1.0)].sum()/max(F.sum(),1e-12)))
    b=np.array(b)
    # PER-INDEX velocity: divide by the actual frame-index gap
    dC=np.linalg.norm(C[1:]-C[:-1],axis=1)/np.maximum(idx[1:]-idx[:-1],1)
    dZ=np.degrees(np.arccos(np.clip(np.einsum('id,id->i',Z[1:],Z[:-1]),-1,1)))/np.maximum(idx[1:]-idx[:-1],1)
    v=np.concatenate([[dC[0]],(dC[1:]+dC[:-1])/2,[dC[-1]]])
    w=np.concatenate([[dZ[0]],(dZ[1:]+dZ[:-1])/2,[dZ[-1]]])
    sp=lambda a,c:np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(c)))[0,1]
    print(f"\n{scene}: n={len(fs)}  gaps med {np.median(np.diff(idx)):.0f}")
    print(f"  velocity/index  med {np.median(v):.5f}  CV {v.std()/v.mean():.3f}  p95/p5 {np.percentile(v,95)/max(np.percentile(v,5),1e-9):.2f}x")
    print(f"  corr(beta_t, trans speed/idx)  r={np.corrcoef(v,b)[0,1]:+.3f}  spearman {sp(v,b):+.3f}")
    print(f"  corr(beta_t, ang speed/idx)    r={np.corrcoef(w,b)[0,1]:+.3f}  spearman {sp(w,b):+.3f}")
    print(f"  velocity series first 12: {np.round(v[:12],5).tolist()}")
    print(f"  autocorr(v,lag1) {np.corrcoef(v[:-1],v[1:])[0,1]:+.3f}  (smooth trajectory -> high; noise -> ~0)")

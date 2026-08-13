import os, sys, numpy as np, cv2
sys.path.insert(0,"/mnt/d/avv/metric_probe"); import mlib
cv2.setNumThreads(6)
GTD="/mnt/d/avv/data/phase1/public_set/{s}/test/images"
SC=[("HCM0181","/mnt/d/avv/prodharness/k4/png"),("HCM0193","/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png"),
    ("HCM0204","/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png"),
    ("hcm0031","/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png"),
    ("hcm0034","/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png")]
def af(i8,f):
    H,W,_=i8.shape; fu=cv2.resize(f,(W,H),interpolation=cv2.INTER_CUBIC)
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    return np.clip(cv2.remap(i8.astype(np.float32)/255.,(xx+fu[...,0]).astype(np.float32),(yy+fu[...,1]).astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)*255.,0,255).astype(np.uint8)
tot={}
for sc,rd in SC:
    F=np.load(f"/mnt/d/avv/fields/{sc}.npy"); gd=GTD.format(s=sc)
    gm={os.path.splitext(f)[0]:f for f in sorted(os.listdir(gd))}
    st=sorted(s for s in gm if os.path.exists(os.path.join(rd,s+".png")))
    st=st[::max(1,len(st)//20)][:20]
    Ds=[];MSE=[]
    for s in st:
        G=mlib.load_u8(os.path.join(gd,gm[s])).astype(np.float32)/255.
        R=af(mlib.load_u8(os.path.join(rd,s+".png")),F).astype(np.float32)/255.
        D=cv2.GaussianBlur(G,(0,0),12.8)-cv2.GaussianBlur(R,(0,0),12.8)
        H,W=D.shape[:2]; Ds.append(cv2.resize(D,(W//8,H//8),interpolation=cv2.INTER_AREA))
        MSE.append(float(((G-R)**2).mean()))
    Ds=np.stack(Ds); N=len(st); mse0=np.mean(MSE)
    red_o=[];red_l=[]
    for i in range(N):
        d=Ds[i].ravel(); l=((Ds.sum(0)-Ds[i])/(N-1)).ravel()
        a=float(d@l/max(l@l,1e-12))                 # MSE-optimal gain, per image, ORACLE-chosen
        red_l.append(a*float(d@l)/d.size)           # MSE removed by best-scaled LOO field
        red_o.append(float(d@d)/d.size)             # MSE removed by full oracle substitution
    ro,rl=np.mean(red_o),np.mean(red_l)
    tot[sc]=(mse0,ro,rl)
    print(f"{sc}: base MSE {mse0:.3e} | oracle LP removes {100*ro/mse0:5.2f}% -> dPSNR {-10*np.log10(1-ro/mse0):+.4f} dB"
          f" | BEST-GAIN LOO field removes {100*rl/mse0:5.3f}% -> dPSNR {-10*np.log10(max(1-rl/mse0,1e-9)):+.4f} dB")
m=np.mean([v[0] for v in tot.values()]);o=np.mean([v[1] for v in tot.values()]);l=np.mean([v[2] for v in tot.values()])
print(f"\nPOOLED: oracle dPSNR {-10*np.log10(1-o/m):+.4f} dB (score {0.6*-10*np.log10(1-o/m):+.4f} from PSNR term)")
print(f"POOLED: MSE-OPTIMAL-GAIN LOO field dPSNR {-10*np.log10(1-l/m):+.4f} dB (score {0.6*-10*np.log10(1-l/m):+.4f}) = {100*l/o:.1f}% of the oracle")

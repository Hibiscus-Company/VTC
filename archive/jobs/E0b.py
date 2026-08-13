"""E0b -- normalise away the texture tautology, then ask what is left."""
import os,numpy as np,cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
D="/mnt/d/avv/data/phase1/private_set2/bonsai/train/images"; R="/mnt/d/avv/blurbound/bonsai/train_png"
st=sorted(f[:-4] for f in os.listdir(R) if f.endswith(".png"))
st=[s for s in st if os.path.exists(f"{D}/{s}.jpg")][::2]
dk=[];tx=[];er=[];rw=[]
for s in st:
    r=np.asarray(Image.open(f"{R}/{s}.png").convert("RGB"),dtype=np.float32)
    g=np.asarray(Image.open(f"{D}/{s}.jpg").convert("RGB"),dtype=np.float32)
    if r.shape!=g.shape: continue
    e=np.abs(r-g).mean(2); gl=g.mean(2); lap=np.abs(cv2.Laplacian(gl,cv2.CV_32F))
    H,W=e.shape; ch,cw=8,12; hs,ws=H//ch,W//cw
    for i in range(ch):
        for j in range(cw):
            sl=(slice(i*hs,(i+1)*hs),slice(j*ws,(j+1)*ws))
            dk.append(gl[sl].mean()); tx.append(lap[sl].mean()); er.append(e[sl].mean()); rw.append(i)
dk=np.array(dk);tx=np.array(tx);er=np.array(er);rw=np.array(rw)
# regress err on texture, look at the RESIDUAL
A=np.vstack([tx,np.ones_like(tx)]).T
coef,_,_,_=np.linalg.lstsq(A,er,rcond=None)
res=er-A@coef
print(f"n={len(er)} cells, {len(st)} frames")
print(f"err = {coef[0]:.4f}*texture + {coef[1]:.4f}   (R^2 {1-res.var()/er.var():.3f})")
print(f"\n=== AFTER removing the texture trend, what predicts the RESIDUAL? ===")
print(f"  corr(residual, GT brightness) = {np.corrcoef(dk,res)[0,1]:+.3f}")
lo=dk<np.percentile(dk,25)
print(f"  darkest 25% (glass): residual {res[lo].mean():+.3f} vs rest {res[~lo].mean():+.3f}   -> glass is {'WORSE' if res[lo].mean()>res[~lo].mean() else 'BETTER'} than its texture predicts")
print(f"\n=== residual by row (0=top ... 7=bottom/table) ===")
for i in range(8):
    m=rw==i; print(f"   row {i}: residual {res[m].mean():+7.3f}   raw err {er[m].mean():6.3f}   texture {tx[m].mean():6.3f}")
print(f"\n=== FRACTION OF TOTAL ERROR BY TEXTURE QUARTILE ===")
q=np.percentile(tx,[25,50,75])
for k,(a,b,lab) in enumerate([(-1,q[0],"Q1 smooth"),(q[0],q[1],"Q2"),(q[1],q[2],"Q3"),(q[2],1e9,"Q4 textured")]):
    m=(tx>a)&(tx<=b); print(f"   {lab:12s} {100*er[m].sum()/er.sum():5.1f}% of total error, {100*m.mean():4.1f}% of cells")

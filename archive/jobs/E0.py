"""E0 -- IN-SAMPLE ERROR MAP. 84.4% of bonsai's held-out LPIPS is an in-sample floor, so this
map locates the thing that actually matters. SINGLE model (not the ensemble mean, which costs
33% of VoL and would blur the map). No training."""
import os,numpy as np,cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
D="/mnt/d/avv/data/phase1/private_set2/bonsai/train/images"
R="/mnt/d/avv/blurbound/bonsai/train_png"
st=sorted(f[:-4] for f in os.listdir(R) if f.endswith(".png"))
st=[s for s in st if os.path.exists(f"{D}/{s}.jpg")]
print(f"n={len(st)} paired train views (single model)")
H=W=None; acc=None; per=[]
darks=[];texs=[];errs=[]
for s in st:
    r=np.asarray(Image.open(f"{R}/{s}.png").convert("RGB"),dtype=np.float32)
    g=np.asarray(Image.open(f"{D}/{s}.jpg").convert("RGB"),dtype=np.float32)
    if r.shape!=g.shape: continue
    e=np.abs(r-g).mean(2)
    if acc is None: H,W=e.shape; acc=np.zeros((H,W),np.float64)
    acc+=e; per.append((s,float(e.mean())))
    gl=g.mean(2)
    lap=np.abs(cv2.Laplacian(gl,cv2.CV_32F))
    ch,cw=8,12; hs,ws=H//ch,W//cw
    for i in range(ch):
        for j in range(cw):
            sl=(slice(i*hs,(i+1)*hs),slice(j*ws,(j+1)*ws))
            darks.append(gl[sl].mean()); texs.append(lap[sl].mean()); errs.append(e[sl].mean())
acc/=len(per)
darks=np.array(darks);texs=np.array(texs);errs=np.array(errs)
print(f"\nmean in-sample |err| = {acc.mean():.3f} levels")
print("\n=== ERROR BY IMAGE THIRD (rows) ===")
for k,(a,b,lab) in enumerate([(0,H//3,"TOP    (canopy/background)"),(H//3,2*H//3,"MIDDLE (plant)"),(2*H//3,H,"BOTTOM (glass table)")]):
    print(f"  {lab:28s} {acc[a:b].mean():7.3f}   ({100*acc[a:b].mean()/acc.mean():5.1f}% of mean)")
print("\n=== 8x12 CELL GRID, top-8 worst cells (row,col) ===")
ch,cw=8,12;hs,ws=H//ch,W//cw
cell=np.array([[acc[i*hs:(i+1)*hs, j*ws:(j+1)*ws].mean() for j in range(cw)] for i in range(ch)])
fl=[(cell[i,j],i,j) for i in range(ch) for j in range(cw)]
for v,i,j in sorted(fl,reverse=True)[:8]: print(f"   row {i} col {j:2d}  err {v:6.3f}")
print(f"\n  worst row mean {cell.max(1).max():.3f} at row {cell.mean(1).argmax()}; best {cell.min():.3f}")
print("\n=== WHAT PREDICTS ERROR (cell level, n={}) ===".format(len(errs)))
print(f"  corr(err, GT brightness) = {np.corrcoef(darks,errs)[0,1]:+.3f}   (glass is DARK -> negative means glass is worse)")
print(f"  corr(err, GT texture)    = {np.corrcoef(texs,errs)[0,1]:+.3f}")
lo=darks<np.percentile(darks,25)
print(f"  darkest 25% of cells: err {errs[lo].mean():.3f} vs rest {errs[~lo].mean():.3f}  ratio {errs[lo].mean()/errs[~lo].mean():.2f}x")
hi=texs>np.percentile(texs,75)
print(f"  most-textured 25%   : err {errs[hi].mean():.3f} vs rest {errs[~hi].mean():.3f}  ratio {errs[hi].mean()/errs[~hi].mean():.2f}x")
print("\n=== TOP-10 WORST FRAMES ===")
for s,v in sorted(per,key=lambda x:-x[1])[:10]: print(f"   {s}  {v:.3f}")
np.save("/mnt/d/avv/E0_errmap.npy",acc)
Image.fromarray(np.clip(acc/max(acc.max(),1e-6)*255,0,255).astype(np.uint8)).save("/mnt/d/avv/E0_errmap.png")
print("\nsaved /mnt/d/avv/E0_errmap.png (normalised) and .npy")

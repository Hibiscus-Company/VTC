"""CALIBRATE THE PROTOCOL THAT KILLED THE CHAIR FIELD.
Protocol B: the SAME single member, the SAME median ds8 field, scored at REAL TEST poses vs REAL
TEST GT. Deliberately matches lovo.py's surface: raw PNG, no ensemble, no restore, no JPEG.
Ratio B/A tells us how badly leave-one-view-out on TRAIN views under-reads a lens field."""
import os, sys, time
import numpy as np, torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.join(HERE,"lens"))
from fieldlib import LooPool, gauss_smooth, upsample, warp
Image.MAX_IMAGE_PIXELS=None
TAG="HCM0181"; MEM="gsplatB9ut"
RD=f"/mnt/d/avv/output/{TAG}_{MEM}/test_poses_renders_png"
GD=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GD)}
stems=sorted(s for s in gt_by if os.path.exists(os.path.join(RD,s+".png")))
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
base=LooPool(cache["s8"]).pooled("median")          # pooled over ALL train views
ARMS=[("none",None)]
for g in (1.0,1.15,1.30,1.45):
    ARMS.append((f"med_ds8_rlan_g{g}", upsample(base,H,W,"cubic")*g))
    if g==1.0:
        ARMS.append(("med_ds8_g1_rlan_g1.0", upsample(gauss_smooth(base,1),H,W,"cubic")))
    if g==1.30:
        ARMS.append(("med_ds8_g1_rlan_g1.30", upsample(gauss_smooth(base,1),H,W,"cubic")*1.30))
acc={a[0]:[0.,0.,0.] for a in ARMS}; per={a[0]:[] for a in ARMS}; t0=time.time()
for n,s in enumerate(stems):
    img=np.asarray(Image.open(os.path.join(RD,s+".png")).convert("RGB"),dtype=np.float32)/255.
    g=torch.from_numpy(np.asarray(Image.open(os.path.join(GD,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    for nm,fu in ARMS:
        x=img if fu is None else np.clip(warp(img,fu,"lanczos"),0,1)
        r=torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P=10*np.log10(1./max(((r-g)**2).mean().item(),1e-12)); S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
        acc[nm][0]+=P; acc[nm][1]+=S; acc[nm][2]+=L
        per[nm].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    if n%15==0: print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s",flush=True)
N=len(stems)
print(f"\nPROTOCOL B -- REAL TEST POSES, REAL TEST GT, single member {MEM}, raw PNG (n={N})")
print(f"{'arm':>24} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs none':>9} {'se':>7}")
for nm,_ in ARMS:
    P,S,L=(v/N for v in acc[nm]); sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    d=np.array(per[nm])-np.array(per["none"])
    print(f"{nm:>24} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d.mean():+9.4f} {d.std(ddof=1)/np.sqrt(N):7.4f}")

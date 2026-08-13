"""WHAT IS MEMBER 9, 10, 11 WORTH? The composition finding measured 5->8; production ships 8.
Rank all 21 HCM0181 variants by solo PSNR (cheap, no LPIPS), then walk the uniform-mean k-curve
in descending quality order through the FULL shipped chain (restore lam=1 -> field x1.30 -> JPEG).
Ensemble is accumulated incrementally so adding a member costs one image load, not k."""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.join(HERE,"lens"))
from energy_restore import restore
from fieldlib import LooPool, upsample, warp
Image.MAX_IMAGE_PIXELS=None
TAG="HCM0181"; GAIN=1.30
ROOT="/mnt/d/avv/output"
VARS=sorted(d.split(f"{TAG}_")[1] for d in os.listdir(ROOT)
            if d.startswith(f"{TAG}_") and os.path.isdir(f"{ROOT}/{d}/test_poses_renders_png"))
D=lambda m:f"{ROOT}/{TAG}_{m}/test_poses_renders_png"
SHIP=dict(quality=100,subsampling=2,optimize=True,progressive=True)
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
gtd=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
stems=sorted(s for s in gt_by if all(os.path.exists(os.path.join(D(m),s+".png")) for m in VARS))
print(f"{len(VARS)} variants, {len(stems)} common stems", flush=True)

# ---- pass 1: solo PSNR (CPU, no LPIPS) on a 15-view subsample -> quality ranking
sub=stems[::max(1,len(stems)//15)][:15]
solo={}
for m in VARS:
    se=0.
    for s in sub:
        a=np.asarray(Image.open(os.path.join(D(m),s+".png")).convert("RGB"),dtype=np.float32)/255.
        g=np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255.
        se+=10*np.log10(1./max(((a-g)**2).mean(),1e-12))
    solo[m]=se/len(sub)
order=sorted(VARS,key=lambda m:-solo[m])
print("\nSOLO PSNR RANKING (n=15)")
for i,m in enumerate(order): print(f"  {i+1:2d}. {m:22s} {solo[m]:7.3f} dB")

# ---- pass 2: k-curve through the full chain
KS=[4,6,8,10,12,14]
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
lens=upsample(LooPool(cache["s8"]).pooled("median"),H,W,"cubic")*GAIN
acc={k:[0.,0.,0.] for k in KS}; per={k:[] for k in KS}; t0=time.time()
for n,s in enumerate(stems):
    mem=[]
    for k in KS:
        while len(mem)<k: mem.append(ld(os.path.join(D(order[len(mem)]),s+".png")))
        ens=torch.stack(mem).mean(0)
        o=restore(ens,mem,1.0,len(mem)).clamp(0,1)[0].permute(1,2,0).numpy()
        x=np.clip(warp(o,lens,"lanczos"),0,1)
        b=io.BytesIO(); Image.fromarray((x*255+0.5).astype(np.uint8)).save(b,"JPEG",**SHIP)
        j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
        r=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
        g=torch.from_numpy(np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P=10*np.log10(1./max(((r-g)**2).mean().item(),1e-12)); S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
        acc[k][0]+=P; acc[k][1]+=S; acc[k][2]+=L
        per[k].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    if n%15==0: print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s",flush=True)
N=len(stems)
print(f"\nK-CURVE, REAL GT, {TAG}, n={N}, full shipped chain, members added best-first")
print(f"{'k':>3} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs k=8':>9} {'marginal':>9}")
prev=None
for k in KS:
    P,S,L=(v/N for v in acc[k]); sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    d8=np.array(per[k])-np.array(per[8])
    marg="" if prev is None else f"{sc-prev:+9.4f}"
    print(f"{k:3d} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d8.mean():+9.4f} {marg:>9}")
    prev=sc

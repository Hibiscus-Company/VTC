"""WHAT IS MEMBER 9..14 WORTH? Same measurement, but with the member Laplacian L0 computed ONCE
per view and reused across every k -- the previous version recomputed 54 pyramids per view."""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.join(HERE,"lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample, warp
Image.MAX_IMAGE_PIXELS=None
TAG="HCM0181"; GAIN=1.30; NLEV=5; WIN=3; CLAMP=4.0; LAM=1.0
ROOT="/mnt/d/avv/output"
ORDER=["gsplatB11ut60k","sh3","m31b_nolpips","m31b_taillpips","gsplatB10ut8M","gsplatB12ut8Ms7",
       "e17visnorm","e15ceil95","gsplatB9ut","gsplatB8pure","gsplatB2","gsplatB4warm",
       "gsplatB1","gsplatB7ppisp2"]
KS=[4,6,8,10,12,14]
D=lambda m:f"{ROOT}/{TAG}_{m}/test_poses_renders_png"
SHIP=dict(quality=100,subsampling=2,optimize=True,progressive=True)
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
gtd=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
stems=sorted(s for s in gt_by if all(os.path.exists(os.path.join(D(m),s+".png")) for m in ORDER))
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
lens=upsample(LooPool(cache["s8"]).pooled("median"),H,W,"cubic")*GAIN
K=_K
acc={k:[0.,0.,0.] for k in KS}; per={k:[] for k in KS}; t0=time.time()
for n,s in enumerate(stems):
    mem=[ld(os.path.join(D(m),s+".png")) for m in ORDER]
    mL0=[lap_pyr(m,NLEV,K)[0][0] for m in mem]              # ONCE per view
    g=torch.from_numpy(np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    for k in KS:
        ens=torch.stack(mem[:k]).mean(0)
        laps,res,sizes=lap_pyr(ens,NLEV,K); L0=laps[0]
        Eb=boxf((L0**2).sum(1,keepdim=True),WIN)
        V=0.0
        for i in range(k): V=V+boxf(((mL0[i]-L0)**2).sum(1,keepdim=True),WIN)
        V=V/k*(k/(k-1.0))
        r=torch.sqrt(1.0+V/(Eb+1e-10)).clamp(max=CLAMP)
        o=lap_recon([L0*(1.0+LAM*(r-1.0))]+laps[1:],res,sizes,K).clamp(0,1)[0].permute(1,2,0).numpy()
        x=np.clip(warp(o,lens,"lanczos"),0,1)
        b=io.BytesIO(); Image.fromarray((x*255+0.5).astype(np.uint8)).save(b,"JPEG",**SHIP)
        j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
        rr=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P=10*np.log10(1./max(((rr-g)**2).mean().item(),1e-12)); S=float(repo_ssim(rr,g)); L=float(vgg(rr*2-1,g*2-1).item())
        acc[k][0]+=P; acc[k][1]+=S; acc[k][2]+=L
        per[k].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    if n%10==0: print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s",flush=True)
N=len(stems)
print(f"\nK-CURVE, REAL GT, {TAG}, n={N}, full shipped chain, members added BEST-FIRST by solo PSNR")
print(f"{'k':>3} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs k=8':>9} {'se':>7} {'marginal/member':>16}")
prev=None
for k in KS:
    P,S,L=(v/N for v in acc[k]); sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    d8=np.array(per[k])-np.array(per[8])
    marg="" if prev is None else f"{(sc-prev)/2:+16.4f}"
    print(f"{k:3d} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d8.mean():+9.4f} {d8.std(ddof=1)/np.sqrt(N):7.4f} {marg}")
    prev=sc

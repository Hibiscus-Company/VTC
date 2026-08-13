"""ADJUDICATE r30 vs r30b. Two claims from the campaign synthesis contradict my own measurements:
  (1) q98/4:4:4 beats the shipped q100/4:2:0 by +0.0507, and lossless PNG beats it by +0.0517.
      My sweep tested q98 with 4:2:0 and 4:4:4 with q100 -- never the cross term. Test it.
  (2) field gain 1.50, not 1.30. I measured 1.45 == 1.30 twice, 1.30 marginally ahead.
Measured at PRODUCTION DEPTH k=10 with the gauss1 field, which is where the encode claim is
supposedly largest. Ensemble + restore hoisted; arms vary ONLY (gain x encode)."""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.join(HERE,"lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, gauss_smooth, upsample, warp
Image.MAX_IMAGE_PIXELS=None
TAG="HCM0181"; NLEV=5; WIN=3; CLAMP=4.0; LAM=1.0; KK=10
ORDER=["gsplatB11ut60k","sh3","m31b_nolpips","m31b_taillpips","gsplatB10ut8M","gsplatB12ut8Ms7",
       "e17visnorm","e15ceil95","gsplatB9ut","gsplatB8pure"]
D=lambda m:f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
gtd=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
stems=sorted(s for s in gt_by if all(os.path.exists(os.path.join(D(m),s+".png")) for m in ORDER))
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
base=gauss_smooth(LooPool(cache["s8"]).pooled("median"),1)
FIELDS={g: upsample(base,H,W,"cubic")*g for g in (1.30,1.40,1.50,1.60)}
# (name, gain, encode) ; encode None = lossless PNG
ARMS=[("g1.30_q100ss2",1.30,dict(format="JPEG",quality=100,subsampling=2,optimize=True,progressive=True)),
      ("g1.30_q98ss0", 1.30,dict(format="JPEG",quality=98, subsampling=0,optimize=True,progressive=True)),
      ("g1.30_PNG",    1.30,None),
      ("g1.40_q98ss0", 1.40,dict(format="JPEG",quality=98, subsampling=0,optimize=True,progressive=True)),
      ("g1.50_q100ss2",1.50,dict(format="JPEG",quality=100,subsampling=2,optimize=True,progressive=True)),
      ("g1.50_q98ss0", 1.50,dict(format="JPEG",quality=98, subsampling=0,optimize=True,progressive=True)),
      ("g1.60_q98ss0", 1.60,dict(format="JPEG",quality=98, subsampling=0,optimize=True,progressive=True))]
acc={a[0]:[0.,0.,0.] for a in ARMS}; per={a[0]:[] for a in ARMS}; byt={a[0]:0 for a in ARMS}
t0=time.time()
for n,s in enumerate(stems):
    mem=[ld(os.path.join(D(m),s+".png")) for m in ORDER]
    mL0=[lap_pyr(m,NLEV,_K)[0][0] for m in mem]
    ens=torch.stack(mem).mean(0)
    laps,res,sizes=lap_pyr(ens,NLEV,_K); L0=laps[0]
    Eb=boxf((L0**2).sum(1,keepdim=True),WIN)
    V=sum(boxf(((mL0[i]-L0)**2).sum(1,keepdim=True),WIN) for i in range(KK))/KK*(KK/(KK-1.0))
    r=torch.sqrt(1.0+V/(Eb+1e-10)).clamp(max=CLAMP)
    o=lap_recon([L0*(1.0+LAM*(r-1.0))]+laps[1:],res,sizes,_K).clamp(0,1)[0].permute(1,2,0).numpy()
    g=torch.from_numpy(np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    warped={}
    for nm,gain,enc in ARMS:
        if gain not in warped:
            warped[gain]=(np.clip(warp(o,FIELDS[gain],"lanczos"),0,1)*255+0.5).astype(np.uint8)
        u8=warped[gain]
        b=io.BytesIO()
        if enc is None: Image.fromarray(u8).save(b,"PNG",optimize=True)
        else:
            kw=dict(enc); f=kw.pop("format"); Image.fromarray(u8).save(b,f,**kw)
        byt[nm]+=b.tell()
        j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
        rr=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P=10*np.log10(1./max(((rr-g)**2).mean().item(),1e-12)); S=float(repo_ssim(rr,g)); L=float(vgg(rr*2-1,g*2-1).item())
        acc[nm][0]+=P; acc[nm][1]+=S; acc[nm][2]+=L
        per[nm].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    if n%10==0: print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s",flush=True)
N=len(stems); B="g1.30_q100ss2"
print(f"\nADJUDICATION, {TAG}, n={N}, k=10 + gauss1 field, real test GT")
print(f"{'arm':>16} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs r30':>9} {'se':>7} {'wins':>7} {'MiB/60':>8}")
for nm,_,_ in ARMS:
    P,S,L=(v/N for v in acc[nm]); sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    d=np.array(per[nm])-np.array(per[B])
    t=d.mean()/(d.std(ddof=1)/np.sqrt(N)) if d.std(ddof=1)>0 else 0.0
    print(f"{nm:>16} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d.mean():+9.4f} {d.std(ddof=1)/np.sqrt(N):7.4f} {int((d>0).sum()):3d}/{N:<3d} {byt[nm]/1048576*60/N:8.2f}  t={t:.2f}")

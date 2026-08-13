"""LEAVE-ONE-OUT + WEIGHTING on the production harness (HCM0181, REAL test GT, full shipped chain).
Same code path as kcurve2.py (which produced the k-curve the campaign acted on): float member mean,
energy_restore lam=1.0, median lens field gain 1.30, JPEG q100/4:2:0.
Arms: uniform-8 baseline; 8 leave-one-out (uniform mean of the other 7); 3 weighting schemes.
Also dumps per-member solo score + GT-free consensus deviation / HF ratio so the LOO delta can be
regressed on GT-free statistics that ARE computable on the private towers."""
import io, os, sys, time, json
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
POOL=["gsplatB11ut60k","sh3","m31b_nolpips","m31b_taillpips","gsplatB10ut8M","gsplatB12ut8Ms7",
      "e17visnorm","e15ceil95"]           # best-first by solo PSNR, == the harness k=8 ensemble
K=len(POOL)
D=lambda m:f"{ROOT}/{TAG}_{m}/test_poses_renders_png"
SHIP=dict(quality=100,subsampling=2,optimize=True,progressive=True)
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev_t="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev_t).eval()
gtd=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
stems=sorted(s for s in gt_by if all(os.path.exists(os.path.join(D(m),s+".png")) for m in POOL))
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
lens=upsample(LooPool(cache["s8"]).pooled("median"),H,W,"cubic")*GAIN
KER=_K

# ---- pre-pass: GT-free consensus deviation + HF ratio on 8 views, to define weights ----
PRE=stems[::max(1,len(stems)//8)][:8]
dev=np.zeros(K); hfr=np.zeros(K)
for s in PRE:
    mem=[ld(os.path.join(D(m),s+".png")) for m in POOL]
    ens=torch.stack(mem).mean(0)
    ehf=float((lap_pyr(ens,NLEV,KER)[0][0]**2).mean().sqrt())
    for i,m in enumerate(mem):
        dev[i]+=float((m-ens).abs().mean())*255.
        hfr[i]+=float((lap_pyr(m,NLEV,KER)[0][0]**2).mean().sqrt())/ehf
dev/=len(PRE); hfr/=len(PRE)
print("GT-free member stats (dev/255 vs uniform-8 consensus, HF ratio):",flush=True)
for i,m in enumerate(POOL): print(f"  {m:>18} dev {dev[i]:.4f}  HF {hfr[i]:.4f}",flush=True)

def wnorm(w):
    w=np.asarray(w,dtype=np.float64); return w/w.sum()
top2dev=list(np.argsort(-dev)[:2])
ARMS=[("uniform8", wnorm(np.ones(K)))]
for i in range(K):
    w=np.ones(K); w[i]=0.0; ARMS.append((f"LOO_{POOL[i]}", wnorm(w)))
w=np.ones(K); w[0]=1.5; w[1]=1.5; ARMS.append(("w1.5_top2solo", wnorm(w)))
w=np.ones(K); w[top2dev[0]]=1.5; w[top2dev[1]]=1.5; ARMS.append(("w1.5_top2dev(=4:1:1 analog)", wnorm(w)))
ARMS.append(("w_prop_1/dev", wnorm(1.0/dev)))
ARMS.append(("w_prop_dev", wnorm(dev)))
print(f"\n{len(ARMS)} arms, n={len(stems)} views\n",flush=True)

def metrics(x_np, g):
    b=io.BytesIO(); Image.fromarray((x_np*255+0.5).astype(np.uint8)).save(b,"JPEG",**SHIP)
    j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
    rr=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev_t)
    with torch.no_grad():
        P=10*np.log10(1./max(((rr-g)**2).mean().item(),1e-12)); S=float(repo_ssim(rr,g)); L=float(vgg(rr*2-1,g*2-1).item())
    return P,S,L

per={a:[] for a,_ in ARMS}; acc={a:np.zeros(3) for a,_ in ARMS}
solo={m:np.zeros(3) for m in POOL}
t0=time.time()
for n,s in enumerate(stems):
    mem=[ld(os.path.join(D(m),s+".png")) for m in POOL]
    mL0=[lap_pyr(m,NLEV,KER)[0][0] for m in mem]
    g=torch.from_numpy(np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev_t)
    if n<20:  # solo member quality on a subset (cheap, only needs ranking)
        for i,m in enumerate(POOL):
            x=np.clip(warp(mem[i][0].permute(1,2,0).numpy(),lens,"lanczos"),0,1)
            solo[m]+=np.array(metrics(x,g))
    for aname,w in ARMS:
        ens=sum(float(w[i])*mem[i] for i in range(K) if w[i]>0)
        laps,res,sizes=lap_pyr(ens,NLEV,KER); L0=laps[0]
        Eb=boxf((L0**2).sum(1,keepdim=True),WIN)
        V=0.0; sw=0.0; nz=0
        for i in range(K):
            if w[i]<=0: continue
            V=V+float(w[i])*boxf(((mL0[i]-L0)**2).sum(1,keepdim=True),WIN); sw+=float(w[i]); nz+=1
        V=V/sw*(nz/(nz-1.0))
        r=torch.sqrt(1.0+V/(Eb+1e-10)).clamp(max=CLAMP)
        o=lap_recon([L0*(1.0+LAM*(r-1.0))]+laps[1:],res,sizes,KER).clamp(0,1)[0].permute(1,2,0).numpy()
        x=np.clip(warp(o,lens,"lanczos"),0,1)
        P,S,L=metrics(x,g)
        acc[aname]+=np.array([P,S,L]); per[aname].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    if n%5==0: print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s",flush=True)

N=len(stems)
print(f"\n== SOLO MEMBER QUALITY (n=20, field+jpeg, no ensemble) ==")
srank={}
for m in POOL:
    P,S,L=solo[m]/20.; sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)); srank[m]=sc
    print(f"  {m:>18} {sc:8.4f}  PSNR {P:7.4f} SSIM {S:.5f} LPIPS {L:.5f}")
base=np.array(per["uniform8"])
print(f"\n== ARMS, {TAG}, n={N}, real test GT, full shipped chain ==")
print(f"{'arm':>30} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs uniform8':>12} {'se':>7} {'t':>7} {'wins':>6}")
out={}
for aname,w in ARMS:
    P,S,L=acc[aname]/N; sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    d=np.array(per[aname])-base; se=d.std(ddof=1)/np.sqrt(N) if aname!="uniform8" else 0.0
    t=d.mean()/se if se>0 else 0.0
    print(f"{aname:>30} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d.mean():+12.4f} {se:7.4f} {t:+7.2f} {int((d>0).sum()):3d}/{N}")
    out[aname]=dict(score=sc,psnr=P,ssim=S,lpips=L,delta=d.mean(),se=se,wins=int((d>0).sum()))
print("\n== does a GT-free stat predict the LOO delta? ==")
loo=np.array([out[f"LOO_{m}"]["delta"] for m in POOL])
sc_solo=np.array([srank[m] for m in POOL])
for nm,v in [("dev",dev),("HFratio",hfr),("solo_score",sc_solo)]:
    c=np.corrcoef(v,loo)[0,1]
    print(f"  corr(LOO_delta, {nm:>10}) = {c:+.3f}")
json.dump(dict(arms=out,dev=dev.tolist(),hf=hfr.tolist(),solo=srank,pool=POOL,n=N),
          open(f"{HERE}/loo_ens_{TAG}.json","w"),indent=1)
print("\nDONE")

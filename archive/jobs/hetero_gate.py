"""THE GATE FOR r34: at FIXED k=8 and MATCHED solo quality, does POOL HETEROGENEITY pay?
Our private pool disagrees at 2.20/255; the harness pool at 4.67. r30 proved we are PAST our
optimum at k=8, so ADDING members loses. The open question is whether SWAPPING a redundant member
for a decorrelated one -- k unchanged -- gains. If yes, training non-UT members is worth 12h GPU.
If no, r31 stands and we stop.
Builds a MIN-diversity and a MAX-diversity 8-subset from the top-14 by solo PSNR (so quality is
matched as far as possible), scores both through the FULL shipped chain, and reports each pool's
heterogeneity in the same units as the 2.20 vs 4.67 figures."""
import io, os, sys, time, itertools
import numpy as np, torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.join(HERE,"lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, gauss_smooth, upsample, warp
Image.MAX_IMAGE_PIXELS=None
TAG="HCM0181"; NLEV=5; WIN=3; CLAMP=4.0; LAM=1.0; K=8
POOL14=["gsplatB11ut60k","sh3","m31b_nolpips","m31b_taillpips","gsplatB10ut8M","gsplatB12ut8Ms7",
        "e17visnorm","e15ceil95","gsplatB9ut","gsplatB8pure","gsplatB2","gsplatB4warm",
        "gsplatB1","gsplatB7ppisp2"]
D=lambda m:f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
SHIP=dict(quality=100,subsampling=2,optimize=True,progressive=True)
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
gtd=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
stems=sorted(s for s in gt_by if all(os.path.exists(os.path.join(D(m),s+".png")) for m in POOL14))

# ---- pairwise distance + solo PSNR on a subsample, to pick the two subsets
sub=stems[::max(1,len(stems)//12)][:12]
imgs={m:[np.asarray(Image.open(os.path.join(D(m),s+".png")).convert("RGB"),dtype=np.float32)/255. for s in sub] for m in POOL14}
gts=[np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255. for s in sub]
solo={m: np.mean([10*np.log10(1./max(((a-g)**2).mean(),1e-12)) for a,g in zip(imgs[m],gts)]) for m in POOL14}
Dm={}
for a,b in itertools.combinations(POOL14,2):
    Dm[(a,b)]=Dm[(b,a)]=np.mean([np.abs(x-y).mean() for x,y in zip(imgs[a],imgs[b])])*255.
def greedy(maximize):
    sel=[max(POOL14,key=lambda m:solo[m])]
    while len(sel)<K:
        cand=[m for m in POOL14 if m not in sel]
        key=lambda m: np.mean([Dm[(m,s)] for s in sel])
        sel.append(max(cand,key=key) if maximize else min(cand,key=key))
    return sel
DIV8, HOMO8 = greedy(True), greedy(False)
def het(sel):
    out=[]
    for i,s in enumerate(sub):
        st=np.stack([imgs[m][i] for m in sel]); out.append(np.abs(st-st.mean(0)).mean()*255.)
    return np.mean(out)
print(f"DIV8  het {het(DIV8):.3f}/255  meanSolo {np.mean([solo[m] for m in DIV8]):.3f} dB  {DIV8}")
print(f"HOMO8 het {het(HOMO8):.3f}/255  meanSolo {np.mean([solo[m] for m in HOMO8]):.3f} dB  {HOMO8}", flush=True)

# ---- score both through the full shipped chain
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
FU=upsample(gauss_smooth(LooPool(cache["s8"]).pooled("median"),1),H,W,"cubic")*1.30
ARMS=[("HOMO8",HOMO8),("DIV8",DIV8)]
acc={n:[0.,0.,0.] for n,_ in ARMS}; per={n:[] for n,_ in ARMS}; t0=time.time()
for n,s in enumerate(stems):
    g=torch.from_numpy(np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    for nm,sel in ARMS:
        mem=[ld(os.path.join(D(m),s+".png")) for m in sel]
        mL0=[lap_pyr(m,NLEV,_K)[0][0] for m in mem]
        ens=torch.stack(mem).mean(0)
        laps,res,sizes=lap_pyr(ens,NLEV,_K); L0=laps[0]
        Eb=boxf((L0**2).sum(1,keepdim=True),WIN)
        V=sum(boxf(((x-L0)**2).sum(1,keepdim=True),WIN) for x in mL0)/K*(K/(K-1.0))
        r=torch.sqrt(1.0+V/(Eb+1e-10)).clamp(max=CLAMP)
        o=lap_recon([L0*(1.0+LAM*(r-1.0))]+laps[1:],res,sizes,_K).clamp(0,1)[0].permute(1,2,0).numpy()
        x=(np.clip(warp(o,FU,"lanczos"),0,1)*255+0.5).astype(np.uint8)
        b=io.BytesIO(); Image.fromarray(x).save(b,"JPEG",**SHIP)
        j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
        rr=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P=10*np.log10(1./max(((rr-g)**2).mean().item(),1e-12)); S=float(repo_ssim(rr,g)); L=float(vgg(rr*2-1,g*2-1).item())
        acc[nm][0]+=P; acc[nm][1]+=S; acc[nm][2]+=L
        per[nm].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    if n%15==0: print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s",flush=True)
N=len(stems)
print(f"\nHETEROGENEITY GATE, {TAG}, n={N}, FIXED k=8, full shipped chain")
print(f"{'arm':>7} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs HOMO8':>10} {'se':>7} {'wins':>7}")
for nm,_ in ARMS:
    P,S,L=(v/N for v in acc[nm]); sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    d=np.array(per[nm])-np.array(per["HOMO8"])
    print(f"{nm:>7} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d.mean():+10.4f} {d.std(ddof=1)/np.sqrt(N):7.4f} {int((d>0).sum()):3d}/{N:<3d}")
print("\nGATE: DIV8 >> HOMO8 at matched quality => swapping for diversity pays => train non-UT (r34).")
print("      Otherwise heterogeneity is not the lever, r34/r35 are dead, and r31 stands.")

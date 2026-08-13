"""If MORE encode fidelity HURTS, where is the interior optimum? Sweep quality DOWN from 100,
and test an explicit chroma low-pass (which 4:2:0 only approximates). Real GT, full r29 chain.
Ensemble+restore+field hoisted; arms differ ONLY in the encode."""
import io, os, sys, time
import numpy as np, torch, cv2
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.join(HERE,"lens"))
from energy_restore import restore
from fieldlib import LooPool, upsample, warp
Image.MAX_IMAGE_PIXELS=None
TAG="HCM0181"; GAIN=1.30
D=lambda m:f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
POOL=["gsplatB9ut","gsplatB10ut8M","gsplatB11ut60k","gsplatB12ut8Ms7"]
BK=dict(optimize=True,progressive=True)
# (name, quality, subsampling, chroma_blur_sigma)
ARMS=[("q100_ss2",100,2,0.0),("q98_ss2",98,2,0.0),("q96_ss2",96,2,0.0),("q93_ss2",93,2,0.0),
      ("q100_cb0.5",100,2,0.5),("q100_cb1.0",100,2,1.0),("q100_cb2.0",100,2,2.0)]
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)

def chroma_lp(u8, sig):
    if sig <= 0: return u8
    y=cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    for c in (1,2): y[...,c]=cv2.GaussianBlur(y[...,c],(0,0),sig)
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB)

from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
gtd=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
stems=sorted(s for s in gt_by if all(os.path.exists(os.path.join(D(m),s+".png")) for m in POOL))
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
lens=upsample(LooPool(cache["s8"]).pooled("median"),H,W,"cubic")*GAIN
acc={a[0]:[0.,0.,0.] for a in ARMS}; byt={a[0]:0 for a in ARMS}; per={a[0]:[] for a in ARMS}
t0=time.time()
for n,s in enumerate(stems):
    mem=[ld(os.path.join(D(m),s+".png")) for m in POOL]
    ens=0.8*torch.stack(mem[:3]).mean(0)+0.2*mem[3]
    o=restore(ens,mem,1.0,len(mem)).clamp(0,1)[0].permute(1,2,0).numpy()
    x=np.clip(warp(o,lens,"lanczos"),0,1)
    u8=(x*255+0.5).astype(np.uint8)
    g=torch.from_numpy(np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    for nm,q,ss,cb in ARMS:
        b=io.BytesIO(); Image.fromarray(chroma_lp(u8,cb)).save(b,"JPEG",quality=q,subsampling=ss,**BK); byt[nm]+=b.tell()
        j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
        r=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P=10*np.log10(1./max(((r-g)**2).mean().item(),1e-12)); S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
        acc[nm][0]+=P; acc[nm][1]+=S; acc[nm][2]+=L
        per[nm].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    if n%15==0: print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s",flush=True)
N=len(stems)
print(f"\nENCODE FIDELITY SWEEP, REAL GT, {TAG}, n={N}, full r29 chain")
print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs ship':>9} {'se':>7} {'wins':>7} {'MiB/60':>8}")
base="q100_ss2"
for nm,_,_,_ in ARMS:
    P,S,L=(v/N for v in acc[nm]); sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    d=np.array(per[nm])-np.array(per[base]); mib=byt[nm]/1048576*60/N
    print(f"{nm:>12} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d.mean():+9.4f} {d.std(ddof=1)/np.sqrt(N):7.4f} {int((d>0).sum()):3d}/{N:<3d} {mib:8.2f}")

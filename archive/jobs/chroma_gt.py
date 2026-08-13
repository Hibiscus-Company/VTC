"""TOWER, REAL GT: does 4:2:2 / 4:4:4 buy score? HCM0181 production harness, full r29 chain.
Chain is hoisted: ensemble+restore+field computed ONCE per view, arms differ only in the encode."""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore
from fieldlib import LooPool, upsample, warp
Image.MAX_IMAGE_PIXELS = None
TAG="HCM0181"; GAIN=1.30
D=lambda m:f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
POOL=["gsplatB9ut","gsplatB10ut8M","gsplatB11ut60k","gsplatB12ut8Ms7"]
ARMS=[("q100_ss2",100,2),("q100_ss1",100,1),("q100_ss0",100,0)]
BK=dict(optimize=True,progressive=True)
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)

from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
gtd=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
stems=sorted(s for s in gt_by if all(os.path.exists(os.path.join(D(m),s+".png")) for m in POOL))
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
lens=upsample(LooPool(cache["s8"]).pooled("median"),H,W,"cubic")*GAIN
acc={a[0]:[0.,0.,0.] for a in ARMS}; byt={a[0]:0 for a in ARMS}; t0=time.time()
per={a[0]:[] for a in ARMS}
for n,s in enumerate(stems):
    mem=[ld(os.path.join(D(m),s+".png")) for m in POOL]
    ens=0.8*torch.stack(mem[:3]).mean(0)+0.2*mem[3]
    o=restore(ens,mem,1.0,len(mem)).clamp(0,1)[0].permute(1,2,0).numpy()
    x=np.clip(warp(o,lens,"lanczos"),0,1)
    im=Image.fromarray((x*255+0.5).astype(np.uint8))
    g=torch.from_numpy(np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    for nm,q,ss in ARMS:
        b=io.BytesIO(); im.save(b,"JPEG",quality=q,subsampling=ss,**BK); byt[nm]+=b.tell()
        j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
        r=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P=10*np.log10(1./max(((r-g)**2).mean().item(),1e-12)); S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
        acc[nm][0]+=P; acc[nm][1]+=S; acc[nm][2]+=L
        per[nm].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    if n%15==0: print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s",flush=True)
N=len(stems)
print(f"\nCHROMA SUBSAMPLING, REAL GT, {TAG}, n={N}, full r29 chain (restore lam=1 -> field x1.30 -> JPEG)")
print(f"{'arm':>10} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs ss2':>9} {'se':>7} {'wins':>6} {'MiB/60':>8}")
base=None
for nm,_,_ in ARMS:
    P,S,L=(v/N for v in acc[nm]); sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    if base is None: base=nm
    d=np.array(per[nm])-np.array(per[base]); mib=byt[nm]/1048576*60/N
    print(f"{nm:>10} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d.mean():+9.4f} {d.std(ddof=1)/np.sqrt(N):7.4f} {int((d>0).sum()):3d}/{N:<3d} {mib:8.2f}")

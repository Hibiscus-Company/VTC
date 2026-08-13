"""GOLD-STANDARD CHECK on the deadband fix: public HCM0181, REAL test GT, FULL shipped chain
(mean -> restore -> median lens field lanczos4 -> JPEG q100/ss2). Same lambda in both arms; the
ONLY difference is whether the ensemble mean is pre-rounded to uint8 before restore.
This is the load-bearing evidence for changing 300 leaderboard-anchored tower files, so it gets
measured here rather than inherited from a sub-agent's 'partial' confidence."""
import os,sys,io,numpy as np,torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE); sys.path.insert(0,HERE+"/lens")
from utils.loss_utils import ssim as repo_ssim
from energy_restore import restore, SHIPPED_JPEG
from lens.fieldlib import LooPool, upsample, warp
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cpu"; torch.set_num_threads(4)
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
TAG="HCM0181"; ROOT="/mnt/d/avv/output"; GAIN=1.30
ORDER=["gsplatB11ut60k","sh3","m31b_nolpips","m31b_taillpips","gsplatB10ut8M","gsplatB12ut8Ms7","e17visnorm","e15ceil95"]
D=lambda m:f"{ROOT}/{TAG}_{m}/test_poses_renders_png"
gtd=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt={os.path.splitext(f)[0]:f"{gtd}/{f}" for f in os.listdir(gtd)}
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
stems=sorted(s for s in gt if all(os.path.exists(f"{D(m)}/{s}.png") for m in ORDER))[:10]
cache=np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H,W=[int(x) for x in cache["HW"]]
lens=upsample(LooPool(cache["s8"]).pooled("median"),H,W,"cubic")*GAIN
def jenc(t):
    a=(t.clamp(0,1)[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    b=io.BytesIO(); Image.fromarray(a).save(b,format="JPEG",**SHIPPED_JPEG)
    return ld(io.BytesIO(b.getvalue()))
def r8(t):
    a=(t.clamp(0,1)[0].permute(1,2,0).numpy()*255+0.5).astype(np.uint8)
    return torch.from_numpy(a.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)
K=len(ORDER); LAMS=[0.25,0.5]
acc={}
for s in stems:
    mem=[ld(f"{D(m)}/{s}.png") for m in ORDER]
    fm=torch.stack(mem).mean(0); rm=r8(fm)
    g=ld(gt[s])
    for l in LAMS:
        for tag,base in (("ROUNDED",rm),("FLOAT",fm)):
            t=restore(base,mem,l,K).clamp(0,1)
            hw=t[0].permute(1,2,0).numpy().astype(np.float32)      # warp() wants HxWx3 numpy
            w_=warp(hw,lens,"lanczos")                              # fieldlib key is "lanczos"
            o=jenc(torch.from_numpy(w_).permute(2,0,1).unsqueeze(0))
            a=acc.setdefault((l,tag),[0.,0.,0.])
            a[0]+=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12)); a[1]+=float(repo_ssim(o,g)); a[2]+=float(vgg(o*2-1,g*2-1).mean())
n=len(stems)
print(f"\nDEADBAND A/B on the public harness, n={n} real test poses, k={K}, full shipped chain\n")
print(f"{'lam':>5} {'arm':>8} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'SCORE':>8} {'FLOAT-ROUNDED':>14}")
for l in LAMS:
    vals={}
    for tag in ("ROUNDED","FLOAT"):
        P,S,L=[x/n for x in acc[(l,tag)]]; vals[tag]=sc(P,S,L)
        print(f"{l:5.2f} {tag:>8} {P:7.3f} {S:7.4f} {L:7.4f} {vals[tag]:8.4f}")
    print(f"{'':5} {'':>8} {'':>7} {'':>7} {'':>7} {'':>8} {vals['FLOAT']-vals['ROUNDED']:+14.4f}")
print("\nOur private towers deliver 0.13-0.58 LSB at lam=1.0, i.e. they sit in the LOW-amplitude")
print("regime -- read the lam=0.25/0.5 rows, not lam=1.0, for the expected private-side recovery.")

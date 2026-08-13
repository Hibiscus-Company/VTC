"""DEADBAND A/B ON REAL HELD-OUT GT (bonsai eval split, 28 frames, shipped encode).
arm ROUNDED = the shipped path: restore reads round(mean) as uint8, so any correction < 0.5 LSB
              is thresholded to zero.
arm FLOAT   = identical operator, but the mean is kept in float32 and rounded ONCE at the end.
Same members, same lambda, same encode -> the only variable is the intermediate rounding."""
import os,sys,io,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from utils.loss_utils import ssim as repo_ssim
from energy_restore import restore, SHIPPED_JPEG
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
ES="/mnt/d/avv/evalsplit/bonsai"; B="/mnt/d/avv/bonsai_eval"
ARMS=[d for d in sorted(os.listdir(B)) if os.path.isdir(f"{B}/{d}/eval_png") and d!="K2_clip_full"]
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
def jenc(t):
    a=(t.clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    b=io.BytesIO(); Image.fromarray(a).save(b,format="JPEG",**SHIPPED_JPEG)
    return ld(io.BytesIO(b.getvalue())).to(t.device)
def r8(t):   # emulate the shipped intermediate uint8 write
    a=(t.clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255+0.5).astype(np.uint8)
    return torch.from_numpy(a.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(t.device)
stems=sorted(s for s in gt if all(os.path.exists(f"{B}/{a}/eval_png/{s}.png") for a in ARMS))
K=len(ARMS); LAM=[0.25,1.0]
acc={}
for st in stems:
    mem=[ld(f"{B}/{a}/eval_png/{st}.png").to(dev) for a in ARMS]
    fm=torch.stack(mem).mean(0)          # float mean
    rm=r8(fm)                             # what the shipped chain actually reads back
    g=ld(gt[st]).to(dev)
    if g.shape!=fm.shape: g=torch.nn.functional.interpolate(g,size=fm.shape[-2:],mode="bilinear",align_corners=False)
    cases={"base(no restore)":fm}
    for l in LAM:
        cases[f"ROUNDED lam={l}"]=restore(rm,mem,l,K)
        cases[f"FLOAT   lam={l}"]=restore(fm,mem,l,K)
    for kname,v in cases.items():
        o=jenc(v); a=acc.setdefault(kname,[0.,0.,0.])
        a[0]+=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12))
        a[1]+=float(repo_ssim(o,g)); a[2]+=float(vgg(o*2-1,g*2-1).mean())
n=len(stems)
print(f"\nBONSAI deadband A/B, n={n} held-out frames, k={K}, shipped encode\n")
print(f"{'arm':>20} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'SCORE':>8} {'d(base)':>8}")
base=None
for kname in ["base(no restore)"]+[f"{p} lam={l}" for l in LAM for p in ("ROUNDED","FLOAT  ")]:
    kk=kname.replace("FLOAT   ","FLOAT   ") if kname in acc else kname
    if kk not in acc:
        kk=[x for x in acc if x.startswith(kname.split()[0]) and kname.split()[-1] in x]
        kk=kk[0] if kk else None
    if kk is None: continue
    P,S,L=[x/n for x in acc[kk]]; s=sc(P,S,L)
    if base is None: base=s
    print(f"{kk:>20} {P:7.3f} {S:7.4f} {L:7.4f} {s:8.3f} {s-base:+8.3f}")
print("\nLB value of a bonsai-only gain g is g/7.")

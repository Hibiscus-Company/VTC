"""SWEEP THE BONSAI ENERGY-RESTORATION LAMBDA ON REAL HELD-OUT GT.
r32 ships lam=0.25 chosen by a boost-matching ARGUMENT; the freeze-day audit measured the
delivered boost at ~0.9% against a 27% target. Here there is no need to argue: the bonsai eval
split has 6 member renders and 28 real held-out GT frames, so the optimum is measurable.
Scored through the shipped encode (JPEG q100/ss2) so the number is production-regime."""
import os,sys,io,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from utils.loss_utils import ssim as repo_ssim
from energy_restore import restore, SHIPPED_JPEG
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
ES="/mnt/d/avv/evalsplit/chair"; B="/mnt/d/avv/chair_eval"
ARMS=[d for d in sorted(os.listdir(B)) if os.path.isdir(f"{B}/{d}/eval_png") and d==d]
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
def jenc(t):
    a=(t.clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    b=io.BytesIO(); Image.fromarray(a).save(b,format="JPEG",**SHIPPED_JPEG)
    return ld(io.BytesIO(b.getvalue())).to(t.device), b.tell()
stems=sorted(s for s in gt if all(os.path.exists(f"{B}/{a}/eval_png/{s}.png") for a in ARMS))
LAMS=[0.0,0.25,0.5,0.75,1.0,1.5,2.0]
K=len(ARMS)
acc={l:[0.,0.,0.,0] for l in LAMS}
for st in stems:
    mem=[ld(f"{B}/{a}/eval_png/{st}.png").to(dev) for a in ARMS]
    ens=torch.stack(mem).mean(0)
    g=ld(gt[st]).to(dev)
    if g.shape!=ens.shape: g=torch.nn.functional.interpolate(g,size=ens.shape[-2:],mode="bilinear",align_corners=False)
    for l in LAMS:
        o=ens if l==0 else restore(ens,mem,l,K)
        o,nb=jenc(o)
        acc[l][0]+=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12))
        acc[l][1]+=float(repo_ssim(o,g)); acc[l][2]+=float(vgg(o*2-1,g*2-1).mean()); acc[l][3]+=nb
n=len(stems)
print(f"CHAIR lambda sweep, n={n} held-out frames, k={K} members, shipped encode\n")
print(f"{'lam':>5} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'SCORE':>8} {'d(lam=0)':>9} {'MB/28':>7}")
base=None
for l in LAMS:
    P,S,L,by=acc[l][0]/n,acc[l][1]/n,acc[l][2]/n,acc[l][3]
    s=sc(P,S,L); base=s if base is None else base
    print(f"{l:5.2f} {P:7.3f} {S:7.4f} {L:7.4f} {s:8.3f} {s-base:+9.3f} {by/1e6:7.2f}")
print("\nr32 ships lam=0.25. LB value of a bonsai gain g is g/7.")

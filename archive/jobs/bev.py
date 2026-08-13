"""Baseline the bonsai eval split with the CURRENT scorer, so ARM B (8M/60k) is comparable.
Also measures the ENSEMBLE of the old starved members -- the thing r32 actually ships."""
import os,sys,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
ES="/mnt/d/avv/evalsplit/bonsai"; B="/mnt/d/avv/bonsai_eval"
gt={os.path.splitext(f)[0]:os.path.join(ES,"eval_gt",f) for f in os.listdir(ES+"/eval_gt")}
ARMS=[d for d in sorted(os.listdir(B)) if os.path.isdir(f"{B}/{d}/eval_png") and d!="K2_clip_full"]
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
def score(getim,stems):
    P=S=L=0.
    for st in stems:
        r=getim(st).to(dev); g=ld(gt[st]).to(dev)
        if r.shape!=g.shape: g=torch.nn.functional.interpolate(g,size=r.shape[-2:],mode="bilinear",align_corners=False)
        P+=10*np.log10(1/max(((r-g)**2).mean().item(),1e-12)); S+=float(repo_ssim(r,g)); L+=float(vgg(r*2-1,g*2-1).mean())
    n=len(stems); return P/n,S/n,L/n
stems=sorted(s for s in gt if all(os.path.exists(f"{B}/{a}/eval_png/{s}.png") for a in ARMS))
print(f"n={len(stems)} common eval frames, {len(ARMS)} arms\n")
print(f"{'arm':>16} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'SCORE':>8}")
for a in ARMS:
    P,S,L=score(lambda st,a=a: ld(f"{B}/{a}/eval_png/{st}.png"),stems)
    print(f"{a:>16} {P:7.3f} {S:7.4f} {L:7.4f} {sc(P,S,L):8.3f}",flush=True)
P,S,L=score(lambda st: torch.stack([ld(f"{B}/{a}/eval_png/{st}.png") for a in ARMS]).mean(0),stems)
print(f"{'MEAN(all '+str(len(ARMS))+')':>16} {P:7.3f} {S:7.4f} {L:7.4f} {sc(P,S,L):8.3f}")
print("\n(17/07 recorded capD 5M/30k = 71.156, LPIPS 0.2595 -- different code, use the rows above as the live baseline)")

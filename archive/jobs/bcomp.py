"""BONSAI COMPOSITION SELECTOR on real held-out GT, through the shipped encode.
The 6-member mean beats the best single on SCORE (+0.11) but LOSES on LPIPS (.2560 vs .2448).
Since LPIPS carries weight 0.4 and bonsai is the scene dragging that term, the ensemble depth
that maximises SCORE on this scene is worth measuring rather than inheriting. Also evaluates the
retrained 8M/60k member (arm NEW) and blends, if it has landed."""
import os,sys,io,itertools,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from utils.loss_utils import ssim as repo_ssim
from energy_restore import SHIPPED_JPEG
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
ES="/mnt/d/avv/evalsplit/bonsai"; B="/mnt/d/avv/bonsai_eval"
OLD=[d for d in sorted(os.listdir(B)) if os.path.isdir(f"{B}/{d}/eval_png") and d!="K2_clip_full"]
NEW="/mnt/d/avv/r33_bonsai/eval_s42/eval_png"
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
def jenc(t):
    a=(t.clamp(0,1)[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    b=io.BytesIO(); Image.fromarray(a).save(b,format="JPEG",**SHIPPED_JPEG)
    return ld(io.BytesIO(b.getvalue()))
paths=lambda a,s: f"{B}/{a}/eval_png/{s}.png"
stems=sorted(s for s in gt if all(os.path.exists(paths(a,s)) for a in OLD))
hasnew=os.path.isdir(NEW) and len([f for f in os.listdir(NEW) if f.endswith(".png")])>=len(stems)
# rank old arms by solo score to define the greedy depth ladder
solo={}
for a in OLD:
    P=S=L=0.
    for s in stems:
        o=jenc(ld(paths(a,s))); g=ld(gt[s])
        if g.shape!=o.shape: g=torch.nn.functional.interpolate(g,size=o.shape[-2:],mode="bilinear",align_corners=False)
        P+=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12)); S+=float(repo_ssim(o,g)); L+=float(vgg(o*2-1,g*2-1).mean())
    n=len(stems); solo[a]=(P/n,S/n,L/n); print(f"solo {a:>16} {P/n:7.3f} {S/n:7.4f} {L/n:7.4f} {sc(P/n,S/n,L/n):8.3f}",flush=True)
order=sorted(OLD,key=lambda a:-sc(*solo[a]))
CASES=[(f"MEAN top-{k}",order[:k]) for k in range(1,len(order)+1)]
if hasnew:
    CASES=[("NEW alone",["__NEW__"]),("NEW+top1",["__NEW__",order[0]]),("NEW+top2",["__NEW__"]+order[:2]),
           ("NEW+all old",["__NEW__"]+order)]+CASES
print(f"\n{'composition':>16} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'SCORE':>8}")
best=None
for name,arms in CASES:
    P=S=L=0.
    for s in stems:
        ims=[ld(f"{NEW}/{s}.png") if a=="__NEW__" else ld(paths(a,s)) for a in arms]
        o=jenc(torch.stack(ims).mean(0)); g=ld(gt[s])
        if g.shape!=o.shape: g=torch.nn.functional.interpolate(g,size=o.shape[-2:],mode="bilinear",align_corners=False)
        P+=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12)); S+=float(repo_ssim(o,g)); L+=float(vgg(o*2-1,g*2-1).mean())
    n=len(stems); v=sc(P/n,S/n,L/n)
    print(f"{name:>16} {P/n:7.3f} {S/n:7.4f} {L/n:7.4f} {v:8.3f}",flush=True)
    if best is None or v>best[1]: best=(name,v)
print(f"\nBEST: {best[0]} = {best[1]:.3f}   (r32 ships the 7-member production mean; LB value = gain/7)")
print(f"new member present: {hasnew}")

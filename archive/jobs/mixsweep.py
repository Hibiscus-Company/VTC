"""How many lam=0.1 members should the bonsai ensemble contain? Answer it on the eval split,
where all arms are trained on train_sub and the 28 holes are genuinely held out.
mix2 (lam0.01 + lam0.1) beat treat2 (both lam0.1) earlier, so pure replacement may not be optimal."""
import os,io,sys,itertools,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None; torch.set_num_threads(4)
vgg=lpips_pkg.LPIPS(net="vgg").eval()
ES="/mnt/d/avv/evalsplit/bonsai"; B="/mnt/d/avv/bonsai_eval"
OLD={"K1":f"{B}/K1_noUT_aa/eval_png","c42":"/mnt/d/avv/r36_shape/sr001seed42/eval_png",
     "eps10":f"{B}/eps10/eval_png","eps20":f"{B}/eps20/eval_png","ppisp":f"{B}/ppisp_pc/eval_png"}
NEW={"n42":"/mnt/d/avv/r36_shape/sr01/eval_png","n42b":"/mnt/d/avv/r36_shape/sr01b/eval_png",
     "n101":"/mnt/d/avv/r38/sr01_s101/eval_png"}
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
SHIP=dict(quality=100,subsampling=2,optimize=True,progressive=True)
def enc(t):
    a=(t.clamp(0,1)[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    b=io.BytesIO(); Image.fromarray(a).save(b,"JPEG",**SHIP); return ld(io.BytesIO(b.getvalue()))
allд={**OLD,**NEW}
st=sorted(s for s in gt if all(os.path.exists(f"{d}/{s}.png") for d in allд.values()))
print(f"n={len(st)} holes | old {list(OLD)} | new {list(NEW)}",flush=True)
def score(keys):
    P=S=L=0.
    for s in st:
        m=torch.stack([ld(f"{allд[k]}/{s}.png") for k in keys]).mean(0)
        o=enc(m); g=ld(gt[s])
        if g.shape!=o.shape: g=torch.nn.functional.interpolate(g,size=o.shape[-2:],mode="bilinear",align_corners=False)
        P+=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12)); S+=float(repo_ssim(o,g)); L+=float(vgg(o*2-1,g*2-1).mean())
    n=len(st); return sc(P/n,S/n,L/n)
CASES=[("5 old (baseline-ish)",list(OLD)),("3 new only",list(NEW)),
       ("5 old + 3 new",list(OLD)+list(NEW)),("2 old + 3 new",["K1","c42"]+list(NEW)),
       ("3 old + 3 new",["K1","c42","eps10"]+list(NEW)),("2 old + 2 new",["K1","c42","n42","n101"])]
best=None
for nm,ks in CASES:
    v=score(ks); print(f"  {nm:22s} k={len(ks)}  {v:.4f}",flush=True)
    if best is None or v>best[1]: best=(nm,v)
print(f"\nBEST: {best[0]} = {best[1]:.4f}")

import os,io,sys,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None; dev="cpu"; torch.set_num_threads(6)
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
ES="/mnt/d/avv/evalsplit/bonsai"
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
SHIP=dict(quality=100,subsampling=2,optimize=True,progressive=True)
def enc(t):
    a=(t.clamp(0,1)[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    b=io.BytesIO(); Image.fromarray(a).save(b,"JPEG",**SHIP); return ld(io.BytesIO(b.getvalue())), b.tell()
PAIRS={"ctrl2_lam001":["/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png","/mnt/d/avv/r36_shape/sr001seed42/eval_png"],
       "treat2_lam01":["/mnt/d/avv/r36_shape/sr01/eval_png","/mnt/d/avv/r36_shape/sr01b/eval_png"]}
st=sorted(s for s in gt if all(os.path.exists(f"{d}/{s}.png") for v in PAIRS.values() for d in v))
print(f"n={len(st)} holes, ship encode q100/ss2",flush=True)
per={}
for name,dirs in PAIRS.items():
    P=S=L=0.;by=0;rows=[]
    for s in st:
        m=torch.stack([ld(f"{d}/{s}.png") for d in dirs]).mean(0)
        o,nb=enc(m); g=ld(gt[s])
        if g.shape!=o.shape: g=torch.nn.functional.interpolate(g,size=o.shape[-2:],mode="bilinear",align_corners=False)
        p=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12));ss=float(repo_ssim(o,g));ll=float(vgg(o*2-1,g*2-1).mean())
        rows.append(sc(p,ss,ll));P+=p;S+=ss;L+=ll;by+=nb
    n=len(st);per[name]=np.array(rows)
    print(f"{name}: PSNR {P/n:7.4f} SSIM {S/n:.4f} LPIPS {L/n:.4f} SCORE {sc(P/n,S/n,L/n):.4f}  {by/1e6:.2f}MB",flush=True)
d=per["treat2_lam01"]-per["ctrl2_lam001"]
t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
print(f"\nPAIRED DELTA = {d.mean():+.4f}  t={t:+.2f}  wins {int((d>0).sum())}/{len(d)}")
print(f"GATE (need >=+0.12 and t>=2.0): {'PASS' if (d.mean()>=0.12 and t>=2.0) else 'FAIL'}")
print(f"LB value if shipped = {d.mean()*0.95/7:+.4f} (before transfer prior x0.6)")

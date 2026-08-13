"""PER-SCENE TRIAGE on TRAIN views (train photos = LEGAL GT).
Train->test transfer is slope 0.80 with a near-constant 2.03 dB gap (EXPERIMENTS.md ~1200),
so per-scene TRAIN score ranks per-scene TEST score. Question: is the private deficit
CONCENTRATED in one broken scene (the only shape that can yield +2.3) or DIFFUSE?"""
import os,sys,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
D="/mnt/d/avv/data/phase1/private_set2"
SC=[("HCM0421","/mnt/d/avv/r2r9/models/HCM0421_ut42/train_png"),
    ("HCM0539","/mnt/d/avv/r2r9/models/HCM0539_ut42/train_png"),
    ("HCM0540","/mnt/d/avv/r2r9/models/HCM0540_ut42/train_png"),
    ("HCM0644","/mnt/d/avv/r2r9/models/HCM0644_ut42/train_png"),
    ("HCM0674","/mnt/d/avv/r2r9/models/HCM0674_ut42/train_png"),
    ("chair","/mnt/d/avv/blurbound/chair/train_png"),
    ("bonsai","/mnt/d/avv/blurbound/bonsai/train_png")]
N=16
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
print(f"{'scene':>9} {'n':>3} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'TRAINscore':>11} {'proj TEST':>10}")
rows=[]
for name,rd in SC:
    gtd=f"{D}/{name}/train/images"
    gt={os.path.splitext(f)[0]:os.path.join(gtd,f) for f in os.listdir(gtd)}
    stems=sorted(s[:-4] for s in os.listdir(rd) if s.endswith(".png") and s[:-4] in gt)
    stems=stems[::max(1,len(stems)//N)][:N]
    P=S=L=0.0
    for s in stems:
        r=ld(os.path.join(rd,s+".png")).to(dev); g=ld(gt[s]).to(dev)
        if r.shape!=g.shape:
            g=torch.nn.functional.interpolate(g,size=r.shape[-2:],mode="bilinear",align_corners=False)
        m=((r-g)**2).mean().item()
        P+=10*np.log10(1.0/max(m,1e-12)); S+=float(repo_ssim(r,g))
        L+=float(vgg(r*2-1,g*2-1).mean())
    n=len(stems); P/=n;S/=n;L/=n
    ts=sc(P,S,L); proj=sc(P-2.03,S,L)   # constant-gap model, PSNR only
    rows.append((name,n,P,S,L,ts,proj))
    print(f"{name:>9} {n:3d} {P:7.3f} {S:7.4f} {L:7.4f} {ts:11.3f} {proj:10.3f}",flush=True)
a=np.array([r[5] for r in rows]); mean=a.mean()
print(f"\nTRAIN-score mean {mean:.3f}   spread {a.max()-a.min():.3f}")
for name,_,_,_,_,ts,_ in rows:
    print(f"   {name:>9} {ts-mean:+7.3f} vs mean   -> lifting it to the mean is worth {max(0,mean-ts)/7:+.4f} LB pts")

"""ADV lens2: build pooled means and score per-image on CPU. usage: ADV2_pool.py TAG:arm1,arm2,..."""
import os,sys,numpy as np,torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
torch.set_num_threads(int(os.environ.get("NT","8")))
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"; B="/mnt/d/avv/bonsai_eval"; TMP="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
vgg=lpips_pkg.LPIPS(net="vgg").to("cpu").eval()
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
stems=sorted(gt_by)
for spec in sys.argv[1:]:
    tag,arms=spec.split(":"); arms=arms.split(",")
    P=S=L=0.;per=[]
    with torch.no_grad():
        for s in stems:
            a=np.mean([np.asarray(Image.open(f"{B}/{x}/eval_png/{s}.png").convert("RGB"),dtype=np.float64) for x in arms],0)
            u=np.clip(a+0.5,0,255).astype(np.uint8)
            r=torch.from_numpy(u.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)
            g=load(os.path.join(GT,gt_by[s]))
            assert r.shape==g.shape
            mse=((r-g)**2).mean().item()
            p=10*np.log10(1/max(mse,1e-12)); ss=float(repo_ssim(r,g)); l=float(vgg(r*2-1,g*2-1).item())
            P+=p;S+=ss;L+=l;per.append(100*(0.4*(1-l)+0.3*ss+0.3*min(p/50,1)))
    n=len(stems);P/=n;S/=n;L/=n
    sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1))
    print(f"{tag:14s} k={len(arms)} n={n} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {sc:.4f}",flush=True)
    np.save(f"{TMP}/ADV2_per_{tag}.npy",np.array(per))

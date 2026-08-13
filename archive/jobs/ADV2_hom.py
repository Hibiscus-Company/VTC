"""HOMOGENEOUS-POOL test: mean of the two same-recipe replicates (K1_noUT_aa 19/07, sr001seed42 today)."""
import os,sys,numpy as np,torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
torch.set_num_threads(6)
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"
D=["/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png","/mnt/d/avv/r36_shape/sr001seed42/eval_png"]
vgg=lpips_pkg.LPIPS(net="vgg").to("cpu").eval()
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
P=S=L=0.;per=[]
with torch.no_grad():
    for s in sorted(gt_by):
        a=np.mean([np.asarray(Image.open(f"{d}/{s}.png").convert("RGB"),dtype=np.float64) for d in D],0)
        r=torch.from_numpy(np.clip(a+0.5,0,255).astype(np.uint8).astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)
        g=torch.from_numpy(np.asarray(Image.open(f"{GT}/{gt_by[s]}").convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
        p=10*np.log10(1/max(((r-g)**2).mean().item(),1e-12));ss=float(repo_ssim(r,g));l=float(vgg(r*2-1,g*2-1).item())
        P+=p;S+=ss;L+=l;per.append(100*(0.4*(1-l)+0.3*ss+0.3*min(p/50,1)))
n=len(per);P/=n;S/=n;L/=n
print(f"hom2(seed-replicate pair) n={n} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1)):.4f}",flush=True)
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV2_per_hom2.npy",np.array(per))
k=np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/AUDIT_per_K1_noUT_aa.npy");d=np.array(per)-k
print("hom2 - K1(best member) = %+.4f  se %.4f  t %+.2f"%(d.mean(),d.std(ddof=1)/np.sqrt(n),d.mean()/(d.std(ddof=1)/np.sqrt(n))))

import os,sys,numpy as np,torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
torch.set_num_threads(os.cpu_count())
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"
dev="cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
for d,tag in [(a.split("=")[0],a.split("=")[1]) for a in sys.argv[1:]]:
    rs=sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in(".png",".jpg",".jpeg"))
    assert {os.path.splitext(f)[0] for f in rs}==set(gt_by), f"{tag}: stem mismatch"
    P=S=L=0.;per=[]
    with torch.no_grad():
        for f in rs:
            s=os.path.splitext(f)[0]
            r=load(os.path.join(d,f));g=load(os.path.join(GT,gt_by[s]))
            mse=((r-g)**2).mean().item()
            p=10*np.log10(1/max(mse,1e-12));ss=float(repo_ssim(r,g));l=float(vgg(r*2-1,g*2-1).item())
            P+=p;S+=ss;L+=l;per.append(100*(0.4*(1-l)+0.3*ss+0.3*min(p/50,1)))
    n=len(rs);P/=n;S/=n;L/=n
    sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1))
    print(f"{tag:16s} n={n} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {sc:.4f}  perimg_sd {np.std(per,ddof=1):.4f} sem {np.std(per,ddof=1)/np.sqrt(n):.4f}",flush=True)
    np.save(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/AUDIT_per_{tag}.npy",np.array(per))

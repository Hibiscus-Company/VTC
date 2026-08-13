"""Test hypothesis: EXPERIMENTS.md:4320's '6-member mean 72.013' = mean + restore(lam=0.25), NO encode."""
import os,sys,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp")
os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
from energy_restore import restore
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
torch.set_num_threads(6)
ES="/mnt/d/avv/evalsplit/bonsai"; B="/mnt/d/avv/bonsai_eval"
vgg=lpips_pkg.LPIPS(net="vgg").to("cpu").eval()
ARMS=[d for d in sorted(os.listdir(B)) if os.path.isdir(f"{B}/{d}/eval_png") and d!="K2_clip_full"]
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
stems=sorted(gt); K=len(ARMS)
LAMS=[0.0,0.25]
acc={l:[0.,0.,0.] for l in LAMS}; per={l:[] for l in LAMS}
with torch.no_grad():
    for st in stems:
        mem=[ld(f"{B}/{a}/eval_png/{st}.png") for a in ARMS]
        ens=torch.stack(mem).mean(0); g=ld(gt[st])
        for l in LAMS:
            o=ens if l==0 else restore(ens,mem,l,K)
            o=o.clamp(0,1)
            p=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12)); s=float(repo_ssim(o,g)); L=float(vgg(o*2-1,g*2-1).mean())
            acc[l][0]+=p; acc[l][1]+=s; acc[l][2]+=L; per[l].append(100*(0.4*(1-L)+0.3*s+0.3*min(p/50,1)))
n=len(stems)
for l in LAMS:
    P,S,L=[x/n for x in acc[l]]
    print(f"lam={l:.2f} FLOAT-mean NO-ENCODE n={n} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1)):.4f}",flush=True)
    np.save(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV2_per_lam{l}.npy",np.array(per[l]))

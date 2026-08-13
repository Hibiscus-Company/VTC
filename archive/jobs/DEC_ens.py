"""DECISION MEMO: ensemble-level (k=2) test of the scale_reg family + sr01 single.
ctrl2  = mean(K1_noUT_aa, sr001seed42)   -- A/A pair, identical args, lam=0.01  (LOWEST possible diversity)
treat2 = mean(sr0.03, sr01)              -- lam=0.03 + lam=0.1                  (HIGHER diversity: biased PRO-treatment)
mix2   = mean(K1_noUT_aa, sr01)          -- add a treated member to a control member
singles: sr01, K1_noUT_aa  (sr001seed42 and sr0.03 already in XREG_paired.npy)
All means built exactly like bar.py: float64 mean, then round to uint8.
"""
import os,sys,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
torch.set_num_threads(20)
ES="/mnt/d/avv/evalsplit/bonsai"
D={"K1":"/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png",
   "c42":"/mnt/d/avv/r36_shape/sr001seed42/eval_png",
   "s003":"/mnt/d/avv/r35_scalereg/sr0.03/eval_png",
   "s01":"/mnt/d/avv/r36_shape/sr01/eval_png"}
COMBOS={"sr01_single":["s01"],"K1_single":["K1"],
        "ctrl2_K1+c42":["K1","c42"],"treat2_s003+s01":["s003","s01"],"mix2_K1+s01":["K1","s01"]}
vgg=lpips_pkg.LPIPS(net="vgg").eval()
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
stems=sorted(gt)
def arr(p): return np.asarray(Image.open(p).convert("RGB"),dtype=np.float64)
def t(a): return torch.from_numpy((a/255.).astype(np.float32)).permute(2,0,1)[None]
per={k:[] for k in COMBOS}; acc={k:np.zeros(3) for k in COMBOS}
with torch.no_grad():
    for i,st in enumerate(stems):
        g=t(arr(gt[st])); raw={k:arr(f"{v}/{st}.png") for k,v in D.items()}
        for name,mem in COMBOS.items():
            a=np.mean([raw[m] for m in mem],0)
            o=t(np.clip(a+0.5,0,255).astype(np.uint8).astype(np.float64))
            P=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12)); S=float(repo_ssim(o,g)); L=float(vgg(o*2-1,g*2-1).item())
            acc[name]+= (P,S,L); per[name].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1)))
        print(f"[{i+1}/{len(stems)}] {st}",flush=True)
n=len(stems)
out={}
for name in COMBOS:
    P,S,L=acc[name]/n; sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1))
    out[name]=per[name]; print(f"{name:>18} n={n} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {sc:.4f}",flush=True)
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/DEC_ens.npy",{k:np.array(v) for k,v in out.items()},allow_pickle=True)
P=lambda k: np.array(per[k])
for a,b in [("ctrl2_K1+c42","treat2_s003+s01"),("ctrl2_K1+c42","mix2_K1+s01"),("K1_single","sr01_single")]:
    d=P(b)-P(a); print(f"PAIRED {b} - {a}: D {d.mean():+.4f} sd {d.std(ddof=1):.4f} se {d.std(ddof=1)/np.sqrt(n):.4f} t {d.mean()/(d.std(ddof=1)/np.sqrt(n)):+.2f} wins {(d>0).sum()}/{n}",flush=True)

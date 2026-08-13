"""Does the scale_reg delta survive the SHIP regime (q100/ss2 progressive JPEG)?
ctrl2 = mean(K1_noUT_aa, sr001seed42) [lam=0.01 A/A pair]  vs  treat2 = mean(sr0.03, sr01)."""
import os,io,sys,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
torch.set_num_threads(20)
ES="/mnt/d/avv/evalsplit/bonsai"
D={"K1":"/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png","c42":"/mnt/d/avv/r36_shape/sr001seed42/eval_png",
   "s003":"/mnt/d/avv/r35_scalereg/sr0.03/eval_png","s01":"/mnt/d/avv/r36_shape/sr01/eval_png"}
COMBOS={"ctrl2_enc":["K1","c42"],"treat2_enc":["s003","s01"]}
vgg=lpips_pkg.LPIPS(net="vgg").eval()
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
stems=sorted(gt)
def arr(p): return np.asarray(Image.open(p).convert("RGB"),dtype=np.float64)
def t(a): return torch.from_numpy((a/255.).astype(np.float32)).permute(2,0,1)[None]
per={k:[] for k in COMBOS}; acc={k:np.zeros(3) for k in COMBOS}; byt={k:0 for k in COMBOS}
with torch.no_grad():
    for i,st in enumerate(stems):
        g=t(arr(gt[st])); raw={k:arr(f"{v}/{st}.png") for k,v in D.items()}
        for name,mem in COMBOS.items():
            a=np.clip(np.mean([raw[m] for m in mem],0)+0.5,0,255).astype(np.uint8)
            buf=io.BytesIO(); Image.fromarray(a).save(buf,"JPEG",quality=100,subsampling=2,progressive=True)
            byt[name]+=buf.tell(); o=t(np.asarray(Image.open(buf).convert("RGB"),dtype=np.float64))
            P=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12)); S=float(repo_ssim(o,g)); L=float(vgg(o*2-1,g*2-1).item())
            acc[name]+=(P,S,L); per[name].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1)))
        print(f"[{i+1}/{len(stems)}]",flush=True)
n=len(stems)
for k in COMBOS:
    P,S,L=acc[k]/n; print(f"{k:>12} n={n} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1)):.4f}  {byt[k]/1e6:.2f} MB/28",flush=True)
d=np.array(per["treat2_enc"])-np.array(per["ctrl2_enc"])
print(f"PAIRED treat2_enc - ctrl2_enc: D {d.mean():+.4f} sd {d.std(ddof=1):.4f} se {d.std(ddof=1)/np.sqrt(n):.4f} t {d.mean()/(d.std(ddof=1)/np.sqrt(n)):+.2f} wins {(d>0).sum()}/{n}")
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/DEC_enc2.npy",{k:np.array(v) for k,v in per.items()},allow_pickle=True)

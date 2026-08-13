# LENS2: ship-level A/B with a NULL control.
# Pool = the shipped 6 (the BAR). Slot swapped = K1_noUT_aa (the member whose recipe the arm
# reruns: identical args, default seed 42).
#   E_bar  : pool as-is
#   E_null : K1 -> sr001seed42   (SAME args, SAME seed, different run  = null treatment)
#   E_arm  : K1 -> sr0.03        (the proposed change)
import os, sys, numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
torch.set_num_threads(max(2,os.cpu_count()//2))
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"; B="/mnt/d/avv/bonsai_eval"
D={"K1_noUT_aa":f"{B}/K1_noUT_aa/eval_png","K4_pC_seed1k":f"{B}/K4_pC_seed1k/eval_png",
   "K4_pC_seed7":f"{B}/K4_pC_seed7/eval_png","eps10":f"{B}/eps10/eval_png",
   "eps20":f"{B}/eps20/eval_png","ppisp_pc":f"{B}/ppisp_pc/eval_png",
   "sr001seed42":"/mnt/d/avv/r36_shape/sr001seed42/eval_png","sr0.03":"/mnt/d/avv/r35_scalereg/sr0.03/eval_png"}
FIVE=["K4_pC_seed1k","K4_pC_seed7","eps10","eps20","ppisp_pc"]
SETS={"E_bar":["K1_noUT_aa"]+FIVE,"E_null":["sr001seed42"]+FIVE,"E_arm":["sr0.03"]+FIVE}
vgg=lpips_pkg.LPIPS(net="vgg").eval()
gtby={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
stems=sorted(gtby)
def ld(p): return np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)
per={k:[] for k in SETS}
with torch.no_grad():
    for s in stems:
        g=torch.from_numpy(ld(os.path.join(GT,gtby[s]))/255.).permute(2,0,1)[None]
        cache={n:ld(f"{D[n]}/{s}.png") for n in D}
        for k,names in SETS.items():
            a=np.clip(np.mean([cache[n] for n in names],axis=0)+0.5,0,255).astype(np.uint8)
            r=torch.from_numpy(a.astype(np.float32)/255.).permute(2,0,1)[None]
            mse=((r-g)**2).mean().item(); P=10*np.log10(1/max(mse,1e-12))
            S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
            per[k].append((P,S,L,100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1))))
        print(s,{k:round(per[k][-1][3],4) for k in SETS},flush=True)
R={k:np.array(v) for k,v in per.items()}
for k in SETS: print("%-7s PSNR %.4f SSIM %.4f LPIPS %.4f SCORE %.4f"%(k,R[k][:,0].mean(),R[k][:,1].mean(),R[k][:,2].mean(),R[k][:,3].mean()),flush=True)
for a,b in [("E_null","E_bar"),("E_arm","E_bar"),("E_arm","E_null")]:
    d=R[a][:,3]-R[b][:,3]; se=d.std(ddof=1)/np.sqrt(len(d))
    print("%-16s mean %+0.4f se %.4f t %+0.2f wins %d/%d"%(a+"-"+b,d.mean(),se,d.mean()/se,(d>0).sum(),len(d)),flush=True)
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/LENS2_swapnull.npy",np.stack([R[k] for k in ["E_bar","E_null","E_arm"]]))

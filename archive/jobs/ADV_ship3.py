"""ADVERSARIAL lens 3/3 -- SHIPPABILITY of the '+0.3615 ensemble edge'.
Production bonsai ensemble (ENSEMBLE_MANIFEST.md) = 6-7 members of ONE recipe (noUT_aa),
seed-diverse only. bar.py's '6-member MEAN = WHAT SHIPS' is 6 DIFFERENT recipes
(K1 noUT_aa / K4 pC x2 seeds / eps2d x2 / ppisp). Measure how much of +0.3615 is
recipe-heterogeneity vs seed noise averaging.
Exact same scoring math as AUDIT_cpu_score.py (reproduces eval_score.py to 1e-4).
"""
import os,sys,itertools,numpy as np,torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
torch.set_num_threads(os.cpu_count())
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"; B="/mnt/d/avv/bonsai_eval"
vgg=lpips_pkg.LPIPS(net="vgg").to("cpu").eval()
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
STEMS=sorted(gt_by)
def arr(a,s): return np.asarray(Image.open(f"{B}/{a}/eval_png/{s}.png").convert("RGB"),dtype=np.float64)
def t(x): return torch.from_numpy((x.astype(np.float32)/255.)).permute(2,0,1).unsqueeze(0)
def score(arms,tag):
    P=S=L=0.;per=[]
    with torch.no_grad():
        for s in STEMS:
            m=np.mean([arr(a,s) for a in arms],0)
            u=np.clip(m+0.5,0,255).astype(np.uint8)          # round ONCE, same as bar.py
            r=t(u); g=t(np.asarray(Image.open(f"{GT}/{gt_by[s]}").convert("RGB")))
            mse=((r-g)**2).mean().item()
            p=10*np.log10(1/max(mse,1e-12)); ss=float(repo_ssim(r,g)); l=float(vgg(r*2-1,g*2-1).item())
            P+=p;S+=ss;L+=l; per.append(100*(0.4*(1-l)+0.3*ss+0.3*min(p/50,1)))
    n=len(STEMS);P/=n;S/=n;L/=n
    sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1))
    print(f"{tag:26s} k={len(arms)} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {sc:.4f}",flush=True)
    np.save(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADVS3_{tag}.npy",np.array(per))
    return sc
ALL=['K1_noUT_aa','K4_pC_seed1k','K4_pC_seed7','eps10','eps20','ppisp_pc']
JOBS=[
 (['K4_pC_seed7','K4_pC_seed1k'],            "SEEDPAIR_K4"),      # same recipe, seed only
 (['eps10','eps20'],                          "NEARPAIR_eps"),     # same family, eps 0.1 vs 0.2
 (['K1_noUT_aa','eps10'],                     "XPAIR_K1_eps10"),   # cross-recipe
 (['K1_noUT_aa','K4_pC_seed7'],               "XPAIR_K1_K4s7"),    # cross-recipe, wide gap
 (['K4_pC_seed1k','K4_pC_seed7','eps10','eps20','ppisp_pc'], "MEAN5_noK1"),  # ens minus best
 (ALL,                                        "MEAN6_control"),
]
for arms,tag in JOBS: score(arms,tag)

"""ADV lens 3/3 part B.
(1) PRODUCTION ANALOGUE: production bonsai ensemble = 6-7 SEEDS of ONE recipe
    (ENSEMBLE_MANIFEST.md: aa42/aa7/aa13/aa101/aa202/aa303, uniform, no field).
    The only seed-only trio we have on this split at the shipped recipe:
      sr001seed42 (r36_shape, BASE+seed42+scale_reg0.01 = shipped recipe verbatim) 71.6953
      K4_pC_seed1k 71.3069 ; K4_pC_seed7 71.2192
    -> k=3 seed-only mean vs its own best single = the edge production actually buys.
(2) ENCODE DIFFERENTIAL: what ships is JPEG q100/ss2. Is the encode cost the same for a
    single as for a mean? If the single loses more, the edge grows in the ship regime.
"""
import os,sys,io,numpy as np,torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
torch.set_num_threads(max(4,os.cpu_count()//2))
SHIPPED_JPEG=dict(quality=100,subsampling=2,optimize=True,progressive=True)
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"; B="/mnt/d/avv/bonsai_eval"
D={a:f"{B}/{a}/eval_png" for a in ['K1_noUT_aa','K4_pC_seed1k','K4_pC_seed7','eps10','eps20','ppisp_pc']}
D['sr001seed42']="/mnt/d/avv/r36_shape/sr001seed42/eval_png"
vgg=lpips_pkg.LPIPS(net="vgg").to("cpu").eval()
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
STEMS=sorted(gt_by)
def t(x): return torch.from_numpy((x.astype(np.float32)/255.)).permute(2,0,1).unsqueeze(0)
def run(arms,tag,enc=False):
    P=S=L=0.;nb=0
    with torch.no_grad():
        for s in STEMS:
            m=np.mean([np.asarray(Image.open(f"{D[a]}/{s}.png").convert("RGB"),dtype=np.float64) for a in arms],0)
            u=np.clip(m+0.5,0,255).astype(np.uint8)
            if enc:
                b=io.BytesIO(); Image.fromarray(u).save(b,format="JPEG",**SHIPPED_JPEG); nb+=b.tell()
                u=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
            r=t(u); g=t(np.asarray(Image.open(f"{GT}/{gt_by[s]}").convert("RGB")))
            mse=((r-g)**2).mean().item()
            P+=10*np.log10(1/max(mse,1e-12)); S+=float(repo_ssim(r,g)); L+=float(vgg(r*2-1,g*2-1).item())
    n=len(STEMS);P/=n;S/=n;L/=n
    print(f"{tag:26s} k={len(arms)} enc={int(enc)} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1)):.4f} MB {nb/1e6:.2f}",flush=True)
ALL=['K1_noUT_aa','K4_pC_seed1k','K4_pC_seed7','eps10','eps20','ppisp_pc']
SEED3=['sr001seed42','K4_pC_seed1k','K4_pC_seed7']
run(SEED3,"SEED3_prod_analogue")
run(['K1_noUT_aa'],"ENC_single_K1",enc=True)
run(ALL,"ENC_MEAN6",enc=True)
run(SEED3,"ENC_SEED3",enc=True)

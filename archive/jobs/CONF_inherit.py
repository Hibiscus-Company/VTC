"""CONFOUND TEST: does the +0.1767 single-model scale_reg gain SURVIVE ensemble averaging?
Matched 1-of-6 swap. ens_ctrl and ens_arm differ ONLY in scale_reg (0.01 vs 0.03), same seed 42,
same recipe, same 5 partner members, same averaging code as bar.py.
First-order: full 6-member replacement ~= 6 x (ens_arm - ens_ctrl).
"""
import os,sys,numpy as np,torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
os.chdir("/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
torch.set_num_threads(os.cpu_count())
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"; B="/mnt/d/avv/bonsai_eval"
ARMS=[a for a in sorted(os.listdir(B)) if os.path.isdir(f"{B}/{a}/eval_png") and a!="K2_clip_full"]
print("ARMS6:",ARMS,flush=True)
SWAP="K4_pC_seed7"                      # shipped-recipe member, replaced by the seed-42 pair
CTRL="/mnt/d/avv/r36_shape/sr001seed42/eval_png"
ARMD="/mnt/d/avv/r35_scalereg/sr0.03/eval_png"
def path(a,s): return f"{B}/{a}/eval_png/{s}.png"
gtby={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
stems=sorted(gtby)
vgg=lpips_pkg.LPIPS(net="vgg").eval()
def load_np(p): return np.asarray(Image.open(p).convert("RGB"),dtype=np.float64)
def to_t(a): return torch.from_numpy((a/255.).astype(np.float32)).permute(2,0,1)[None]
def sc(r,g):
    mse=((r-g)**2).mean().item(); P=10*np.log10(1/max(mse,1e-12))
    S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
    return P,S,L,100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1))
res={k:[] for k in("base","ctrl","arm")}
proj=[]
with torch.no_grad():
  for s in stems:
    g8=load_np(os.path.join(GT,gtby[s])); g=to_t(g8)
    mem={a:load_np(path(a,s)) for a in ARMS}
    c=load_np(f"{CTRL}/{s}.png"); a3=load_np(f"{ARMD}/{s}.png")
    def mk(d):
        m=np.mean(list(d.values()),0); return to_t(np.clip(m+0.5,0,255).astype(np.uint8).astype(np.float64))
    db=dict(mem); res["base"].append(sc(mk(db),g))
    dc=dict(mem); dc[SWAP]=c;  res["ctrl"].append(sc(mk(dc),g))
    da=dict(mem); da[SWAP]=a3; res["arm"].append(sc(mk(da),g))
    # geometry: is the treatment delta aligned with the ensemble's remaining error?
    ensc=np.mean(list(dc.values()),0); d=(a3-c); e=(ensc-g8)
    proj.append([float((d*e).sum()/max((e*e).sum(),1e-9)), float(np.sqrt((d*d).mean())), float(np.sqrt((e*e).mean()))])
    print(s,"base %.4f ctrl %.4f arm %.4f  d %+0.4f"%(res["base"][-1][3],res["ctrl"][-1][3],res["arm"][-1][3],res["arm"][-1][3]-res["ctrl"][-1][3]),flush=True)
A={k:np.array(v) for k,v in res.items()}
for k in("base","ctrl","arm"):
    v=A[k]; L,S,P=v[:,2].mean(),v[:,1].mean(),v[:,0].mean()
    print("\n%-5s PSNR %.4f SSIM %.4f LPIPS %.4f SCORE %.4f"%(k,P,S,L,100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1))),flush=True)
d=A["arm"][:,3]-A["ctrl"][:,3]; n=len(d); se=d.std(ddof=1)/np.sqrt(n)
print("\nMARGINAL 1-of-6 swap: %+0.4f  sd %.4f  se %.4f  t %.2f  wins %d/%d"%(d.mean(),d.std(ddof=1),se,d.mean()/se,(d>0).sum(),n))
print("EXTRAPOLATED full 6-member replacement (6x marginal): %+0.4f  [95%% %+0.4f, %+0.4f]"%(6*d.mean(),6*(d.mean()-1.96*se),6*(d.mean()+1.96*se)))
print("IMPLIED INHERITANCE vs single gain 0.1767: %.3f  [%.3f, %.3f]"%(6*d.mean()/0.1767,6*(d.mean()-1.96*se)/0.1767,6*(d.mean()+1.96*se)/0.1767))
p=np.array(proj); print("\nprojection of (arm-ctrl) onto ensemble residual: mean %.4f  rms_delta %.3f  rms_ensresid %.3f"%(p[:,0].mean(),p[:,1].mean(),p[:,2].mean()))
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/CONF_inherit.npy",np.stack([A["base"],A["ctrl"],A["arm"]]))

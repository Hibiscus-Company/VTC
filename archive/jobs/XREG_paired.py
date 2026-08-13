import os,sys,numpy as np,torch
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
torch.set_num_threads(8)
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"
A="/mnt/d/avv/r36_shape/sr001seed42/eval_png"   # control  scale_reg 0.01 seed42
B="/mnt/d/avv/r35_scalereg/sr0.03/eval_png"     # arm      scale_reg 0.03 seed42
vgg=lpips_pkg.LPIPS(net="vgg").eval()
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1)[None]
gtby={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
rows=[]
with torch.no_grad():
  for s in sorted(gtby):
    g=load(os.path.join(GT,gtby[s]))
    v=[]
    for d in (A,B):
        r=load(os.path.join(d,s+".png"))
        mse=((r-g)**2).mean().item(); P=10*np.log10(1/max(mse,1e-12))
        S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
        v.append((P,S,L,100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1))))
    rows.append((s,)+v[0]+v[1]); print(s,"ctrl %.4f arm %.4f d %+0.4f"%(v[0][3],v[1][3],v[1][3]-v[0][3]),flush=True)
r=np.array([x[1:] for x in rows],dtype=float)
d=r[:,7]-r[:,3]
print("\nn=%d  ctrl mean %.4f  arm mean %.4f  DELTA %+0.4f"%(len(d),r[:,3].mean(),r[:,7].mean(),d.mean()))
print("per-image d: mean %+0.4f  sd %.4f  se %.4f  t %.2f  wins %d/%d  min %+0.3f max %+0.3f"%(
   d.mean(),d.std(ddof=1),d.std(ddof=1)/np.sqrt(len(d)),d.mean()/(d.std(ddof=1)/np.sqrt(len(d))),(d>0).sum(),len(d),d.min(),d.max()))
for i,nm in enumerate(["PSNR","SSIM","LPIPS"]):
    dd=r[:,4+i]-r[:,i]; print("  d%-6s %+0.5f  (t %.2f)"%(nm,dd.mean(),dd.mean()/(dd.std(ddof=1)/np.sqrt(len(dd)))))
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/XREG_paired.npy",r)

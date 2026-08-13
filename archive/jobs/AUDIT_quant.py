import sys, os, io, numpy as np, torch
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp")
sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp/lens")
from PIL import Image
from energy_restore import restore, load, SHIPPED_JPEG
from utils.loss_utils import ssim as repo_ssim
from fieldlib import LooPool, upsample, warp
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
tag=sys.argv[1] if len(sys.argv)>1 else "HCM0181"
LAM=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
NIM=int(sys.argv[3]) if len(sys.argv)>3 else 30
MEM=[f"/mnt/d/avv/output/{tag}_gsplatB9ut/test_poses_renders_png",
     f"/mnt/d/avv/output/{tag}_gsplatB10ut8M/test_poses_renders_png",
     f"/mnt/d/avv/output/{tag}_gsplatB11ut60k/test_poses_renders_png",
     f"/mnt/d/avv/output/{tag}_gsplatB12ut8Ms7/test_poses_renders_png"]
MEM=[d for d in MEM if os.path.isdir(d)]
import glob
if len(MEM)<3:
    MEM=sorted(glob.glob(f"/mnt/d/avv/output/{tag}_*ut*/test_poses_renders_png"))[:4]
print("MEMBERS:",MEM)
gtd=f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
stems=sorted(s for s in gt_by if all(os.path.exists(os.path.join(d,s+".png")) for d in MEM))[:NIM]
cache=np.load(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/lens/cache/pub_{tag}.npz")
lens=upsample(LooPool(cache["s8"]).pooled("median"),*[int(x) for x in cache["HW"]],"cubic")
def q8(t): return torch.round(t.clamp(0,1)*255.0)/255.0
arms=["A_base","B_prod_2round","C_float","D_dither_mean_only","E_prod_but_field_float"]
acc={a:[0.,0.,0.] for a in arms}
DS=[0.,0.,0.]
for c,s in enumerate(stems):
    mem=[load(os.path.join(d,s+".png"),dev) for d in MEM]
    ens=torch.stack(mem).mean(0)
    ens_q=q8(ens)
    g=torch.from_numpy(np.asarray(Image.open(os.path.join(gtd,gt_by[s])).convert("RGB"),np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    outs={}
    outs["A_base"]=ens_q                                   # production base: rounded mean
    outs["B_prod_2round"]=q8(restore(ens_q,mem,LAM,len(mem),3))   # SHIPPED: round mean, round restore
    outs["C_float"]=restore(ens,mem,LAM,len(mem),3).clamp(0,1)    # validate harness: no intermediate round
    outs["D_dither_mean_only"]=q8(restore(ens,mem,LAM,len(mem),3))# keep mean float, round once after restore
    outs["E_prod_but_field_float"]=restore(ens_q,mem,LAM,len(mem),3).clamp(0,1) # round mean only
    dc=(outs["C_float"]-ens)[0].cpu().numpy()*255.0
    dq=(outs["B_prod_2round"]-ens_q)[0].cpu().numpy()*255.0
    DS[0]+=np.abs(dc).mean(); DS[1]+=np.abs(dq).mean(); DS[2]+=float((np.round(dc)*dc).sum()/max((dc*dc).sum(),1e-9))
    for a in arms:
        x=outs[a].clamp(0,1)[0].permute(1,2,0).cpu().numpy()
        x=np.clip(warp(x,lens,"lanczos"),0,1)
        b=io.BytesIO(); Image.fromarray((x*255+0.5).astype(np.uint8)).save(b,"JPEG",**SHIPPED_JPEG)
        j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),np.float32)/255.
        r=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad():
            acc[a][0]+=10*np.log10(1./max(((r-g)**2).mean().item(),1e-12))
            acc[a][1]+=float(repo_ssim(r,g)); acc[a][2]+=float(vgg(r*2-1,g*2-1).item())
    if c%10==0: print(f"  {c}/{len(stems)}",flush=True)
n=len(stems); print(f"DELTA: continuous mean|d| {DS[0]/n:.4f} LSB  delivered {DS[1]/n:.4f} LSB  alpha {DS[2]/n:.4f}")
print(f"\n{tag} lam={LAM} n={n}")
base=None
for a in arms:
    P,S,L=(x/n for x in acc[a]); sc=100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
    if base is None: base=sc
    print(f"{a:>24} {sc:9.4f} PSNR {P:8.4f} SSIM {S:7.5f} LPIPS {L:8.5f}  vs base {sc-base:+8.4f}")

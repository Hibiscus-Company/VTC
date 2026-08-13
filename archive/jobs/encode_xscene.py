"""DID THE ENCODE FINDING EVER HOLD ON MORE THAN ONE SCENE? r30c lost 0.0049 where I predicted
+0.06..+0.10, and the encode ladder was measured on HCM0181 ALONE (n=60 VIEWS, n=1 SCENE).
The encode is a pure post-process, so it can be cross-validated on all five public towers using
their single renders + real test GT. Same field construction as shipped (median ds8, gauss1, x1.30)."""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.join(HERE,"lens"))
from fieldlib import LooPool, gauss_smooth, upsample, warp
Image.MAX_IMAGE_PIXELS=None
TOWERS=["HCM0181","HCM0193","HCM0204","hcm0031","hcm0034"]
ARMS=[("q100ss2",dict(quality=100,subsampling=2,optimize=True,progressive=True)),
      ("q99ss0", dict(quality=99, subsampling=0,optimize=True,progressive=True)),
      ("q98ss0", dict(quality=98, subsampling=0,optimize=True,progressive=True))]
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
rows={}
for TAG in TOWERS:
    RD=f"/mnt/d/avv/output/{TAG}_gsplatB9ut/test_poses_renders_png"
    GD=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    cp=f"{HERE}/lens/cache/pub_{TAG}.npz"
    if not (os.path.isdir(RD) and os.path.isdir(GD) and os.path.exists(cp)):
        print(f"  skip {TAG} (missing)"); continue
    gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GD)}
    stems=sorted(s for s in gt_by if os.path.exists(os.path.join(RD,s+".png")))
    cache=np.load(cp); H,W=[int(x) for x in cache["HW"]]
    FU=upsample(gauss_smooth(LooPool(cache["s8"]).pooled("median"),1),H,W,"cubic")*1.30
    acc={a[0]:[0.,0.,0.] for a in ARMS}; per={a[0]:[] for a in ARMS}; t0=time.time()
    for n,s in enumerate(stems):
        img=np.asarray(Image.open(os.path.join(RD,s+".png")).convert("RGB"),dtype=np.float32)/255.
        u8=(np.clip(warp(img,FU,"lanczos"),0,1)*255+0.5).astype(np.uint8)
        g=torch.from_numpy(np.asarray(Image.open(os.path.join(GD,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
        for nm,kw in ARMS:
            b=io.BytesIO(); Image.fromarray(u8).save(b,"JPEG",**kw)
            j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
            r=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P=10*np.log10(1./max(((r-g)**2).mean().item(),1e-12)); S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
            acc[nm][0]+=P; acc[nm][1]+=S; acc[nm][2]+=L
            per[nm].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    N=len(stems); rows[TAG]={}
    for nm,_ in ARMS:
        P,S,L=(v/N for v in acc[nm])
        d=np.array(per[nm])-np.array(per["q100ss2"])
        rows[TAG][nm]=(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)),d.mean(),d.std(ddof=1)/np.sqrt(N),int((d>0).sum()),N,P,S,L)
    print(f"  {TAG} done n={N} {time.time()-t0:.0f}s",flush=True)
print("\nENCODE CROSS-SCENE CHECK -- single member + shipped field, real test GT")
print(f"{'scene':>9} {'arm':>9} {'SCORE':>9} {'vs q100ss2':>11} {'se':>7} {'wins':>7} {'dPSNR':>8} {'dSSIM':>9} {'dLPIPS':>9}")
for TAG in rows:
    b=rows[TAG]["q100ss2"]
    for nm,_ in ARMS:
        sc,dm,se,w,N,P,S,L=rows[TAG][nm]
        print(f"{TAG:>9} {nm:>9} {sc:9.4f} {dm:+11.4f} {se:7.4f} {w:3d}/{N:<3d} {P-b[5]:+8.4f} {S-b[6]:+9.5f} {L-b[7]:+9.5f}")
for nm,_ in ARMS:
    if nm=="q100ss2": continue
    ds=[rows[t][nm][1] for t in rows]
    print(f"\n{nm}: 5-scene mean {np.mean(ds):+.4f}   per-scene {[round(x,4) for x in ds]}"
          f"   scenes positive {sum(1 for x in ds if x>0)}/{len(ds)}")

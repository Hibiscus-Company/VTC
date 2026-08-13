"""CROSS-VALIDATE FIELD SMOOTHING ON 5 TOWERS before shipping it -- the standard r30c taught me.
gauss1 measured +0.0106 on HCM0181 ALONE. Field warps are PER-IMAGE operators independent of
ensemble composition, so unlike the encode they should transfer -- but that is a hypothesis until
measured on more than one scene. Single-member renders + real test GT, gain 1.30 throughout."""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.join(HERE,"lens"))
from fieldlib import LooPool, gauss_smooth, upsample, warp
Image.MAX_IMAGE_PIXELS=None
TOWERS=["HCM0181","HCM0193","HCM0204","hcm0031","hcm0034"]
SHIP=dict(quality=100,subsampling=2,optimize=True,progressive=True)
SIGS=[("plain",0),("gauss1",1),("gauss2",2)]
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev="cuda" if torch.cuda.is_available() else "cpu"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
rows={}
for TAG in TOWERS:
    RD=f"/mnt/d/avv/output/{TAG}_gsplatB9ut/test_poses_renders_png"
    GD=f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    cp=f"{HERE}/lens/cache/pub_{TAG}.npz"
    if not (os.path.isdir(RD) and os.path.isdir(GD) and os.path.exists(cp)): continue
    gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GD)}
    stems=sorted(s for s in gt_by if os.path.exists(os.path.join(RD,s+".png")))
    cache=np.load(cp); H,W=[int(x) for x in cache["HW"]]
    base=LooPool(cache["s8"]).pooled("median")
    FU={nm: upsample(gauss_smooth(base,s) if s>0 else base,H,W,"cubic")*1.30 for nm,s in SIGS}
    acc={nm:[0.,0.,0.] for nm,_ in SIGS}; per={nm:[] for nm,_ in SIGS}; t0=time.time()
    for s in stems:
        img=np.asarray(Image.open(os.path.join(RD,s+".png")).convert("RGB"),dtype=np.float32)/255.
        g=torch.from_numpy(np.asarray(Image.open(os.path.join(GD,gt_by[s])).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
        for nm,_ in SIGS:
            x=(np.clip(warp(img,FU[nm],"lanczos"),0,1)*255+0.5).astype(np.uint8)
            b=io.BytesIO(); Image.fromarray(x).save(b,"JPEG",**SHIP)
            j=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),dtype=np.float32)/255.
            r=torch.from_numpy(np.ascontiguousarray(j)).permute(2,0,1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P=10*np.log10(1./max(((r-g)**2).mean().item(),1e-12)); S=float(repo_ssim(r,g)); L=float(vgg(r*2-1,g*2-1).item())
            acc[nm][0]+=P; acc[nm][1]+=S; acc[nm][2]+=L
            per[nm].append(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)))
    N=len(stems); rows[TAG]={}
    for nm,_ in SIGS:
        P,S,L=(v/N for v in acc[nm]); d=np.array(per[nm])-np.array(per["plain"])
        rows[TAG][nm]=(100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.)),d.mean(),d.std(ddof=1)/np.sqrt(N),int((d>0).sum()),N)
    print(f"  {TAG} n={N} {time.time()-t0:.0f}s",flush=True)
print("\nFIELD SMOOTHING CROSS-SCENE -- single member, gain 1.30, shipped encode, real test GT")
print(f"{'scene':>9} {'arm':>8} {'SCORE':>9} {'vs plain':>10} {'se':>7} {'wins':>8}")
for TAG in rows:
    for nm,_ in SIGS:
        sc,dm,se,w,N=rows[TAG][nm]
        print(f"{TAG:>9} {nm:>8} {sc:9.4f} {dm:+10.4f} {se:7.4f} {w:3d}/{N:<3d}")
for nm,_ in SIGS:
    if nm=="plain": continue
    ds=[rows[t][nm][1] for t in rows]
    print(f"\n{nm}: 5-scene mean {np.mean(ds):+.4f}  per-scene {[round(x,4) for x in ds]}"
          f"  positive {sum(1 for x in ds if x>0)}/{len(ds)}")

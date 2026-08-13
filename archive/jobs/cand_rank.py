"""GT-free quality check on the private tower candidates. The shipped 8-member ensemble is the best
available estimate of truth, so a WORSE member deviates from it more. Reported alongside each
candidate's high-frequency energy (Laplacian L0 RMS) relative to the shipped ensemble -- the
k-curve says LPIPS turns over past k=10, so a member that is SOFTER than the pool is the one that
does the damage."""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp")
import torch
from lapfuse import lap_pyr, _K
Image.MAX_IMAGE_PIXELS=None
TOWERS=["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"]
CAND=[("s2g_champA", "/mnt/d/avv/output_s2gates/{T}_champA/test_png"),
      ("s2g_memB",   "/mnt/d/avv/output_s2gates/{T}_memB/test_png"),
      ("s2g_memC",   "/mnt/d/avv/output_s2gates/{T}_memC/test_png"),
      ("r17_ema999", "/mnt/d/avv/r17/{T}_ut7_ema999/test_png"),
      ("r2r8_ut42",  "/mnt/d/avv/r2r8/models/{T}_ut42/test_png"),
      ("r2r8_ut7",   "/mnt/d/avv/r2r8/models/{T}_ut7/test_png")]
SHIPPED=[("r2r9_ut7","/mnt/d/avv/r2r9/models/{T}_ut7/test_png"),
         ("r2r9_ut42","/mnt/d/avv/r2r9/models/{T}_ut42/test_png"),
         ("seed101","/mnt/d/avv/r22_seed101/{T}/test_png"),
         ("mip555","/mnt/d/avv/r25_mip3d/{T}/test_png")]
ENS="/mnt/d/avv/r29/tower_ens/{T}/png_ens"
NS=8
def ld(p): return np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.
def hf(a):
    t=torch.from_numpy(a).permute(2,0,1).unsqueeze(0)
    return float((lap_pyr(t,5,_K)[0][0]**2).mean().sqrt())
res={}
for T in TOWERS:
    ed=ENS.format(T=T)
    stems=sorted(f for f in os.listdir(ed) if f.endswith(".png"))[::max(1,60//NS)][:NS]
    ens=[ld(os.path.join(ed,s)) for s in stems]
    ehf=np.mean([hf(e) for e in ens])
    for nm,pat in SHIPPED+CAND:
        d=pat.format(T=T)
        if not os.path.isdir(d): continue
        dev=[]; h=[]
        for s,e in zip(stems,ens):
            p=os.path.join(d,s)
            if not os.path.exists(p): break
            a=ld(p); dev.append(np.abs(a-e).mean()*255.); h.append(hf(a))
        if not dev: continue
        res.setdefault(nm,[]).append((np.mean(dev), np.mean(h)/ehf))
print(f"{'member':>12} {'role':>9} {'dev/255':>9} {'HF vs ens':>10}   (lower dev = closer to consensus; HF<1 = softer)")
ship_names={n for n,_ in SHIPPED}
for nm,_ in SHIPPED+CAND:
    if nm not in res: continue
    dv=np.mean([x[0] for x in res[nm]]); hh=np.mean([x[1] for x in res[nm]])
    print(f"{nm:>12} {'SHIPPED' if nm in ship_names else 'candidate':>9} {dv:9.4f} {hh:10.3f}")

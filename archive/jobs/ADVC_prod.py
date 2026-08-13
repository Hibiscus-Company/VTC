"""Production bonsai member disagreement vs eval-arm disagreement (free, no scorer)."""
import os,itertools,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
M=["/mnt/d/avv/r14/bonsai_aa42/test_png","/mnt/d/avv/r14/bonsai_aa7/test_png","/mnt/d/avv/r14/bonsai_aa13/test_png",
   "/mnt/d/avv/r24_bonsai/aa101/test_png","/mnt/d/avv/r24_bonsai/aa202/test_png","/mnt/d/avv/r24_bonsai/aa303/test_png"]
for d in M: print(d, "OK" if os.path.isdir(d) else "MISSING", len(os.listdir(d)) if os.path.isdir(d) else "")
sts=sorted(f for f in os.listdir(M[0]) if f.lower().endswith((".png",".jpg")))[:10]
dis={p:[] for p in itertools.combinations(range(6),2)}
hf=[[] for _ in range(6)]; hfm=[]
def grad(a): return float(np.sqrt(((np.diff(a,0 if 0 else 0,axis=0)**2).mean()+(np.diff(a,axis=1)**2).mean())/2)) if False else float(np.sqrt(((np.diff(a,axis=0)**2).mean()+(np.diff(a,axis=1)**2).mean())/2))
for s in sts:
    A=[np.asarray(Image.open(os.path.join(d,s)).convert("RGB"),dtype=np.float64) for d in M]
    for p in dis: dis[p].append(float(np.sqrt(((A[p[0]]-A[p[1]])**2).mean())))
    for i in range(6): hf[i].append(grad(A[i].mean(2)))
    hfm.append(grad(np.mean(A,0).mean(2)))
v=[np.mean(x) for x in dis.values()]
print(f"PRODUCTION bonsai 6 members (seeds 42/7/13/101/202/303), n={len(sts)} test frames")
print(f"  pairwise RMS disagreement: mean {np.mean(v):.3f} LSB  min {np.min(v):.3f}  max {np.max(v):.3f}")
print(f"  HF(member) {np.mean(hf):.3f}  HF(mean6) {np.mean(hfm):.3f}")
print(f"  EVAL-ARM pool for comparison: same-recipe seed pair 7.324, cross-recipe mean 7.485")
amb=np.mean(v)**2*(6*5)/(2*36); print(f"  ambiguity (PSNR-gain budget) = {amb:.2f} LSB^2   [eval pool = {7.485**2*30/72:.2f}]")

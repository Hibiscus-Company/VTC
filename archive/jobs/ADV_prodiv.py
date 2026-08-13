import os,itertools,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
PROD=["/mnt/d/avv/r14/bonsai_aa42/test_png","/mnt/d/avv/r14/bonsai_aa7/test_png",
      "/mnt/d/avv/r14/bonsai_aa13/test_png","/mnt/d/avv/r24_bonsai/aa101/test_png",
      "/mnt/d/avv/r24_bonsai/aa202/test_png","/mnt/d/avv/r24_bonsai/aa303/test_png"]
for p in PROD: print(p, os.path.isdir(p), len(os.listdir(p)) if os.path.isdir(p) else 0)
PROD=[p for p in PROD if os.path.isdir(p)]
if len(PROD)<2: raise SystemExit("missing production members")
st=sorted(os.listdir(PROD[0]))[:10]
A=[np.stack([np.asarray(Image.open(f"{d}/{s}").convert("RGB"),dtype=np.float32) for s in st]) for d in PROD]
r=[float(np.sqrt(((A[i]-A[j])**2).mean())) for i,j in itertools.combinations(range(len(A)),2)]
sd=float(np.stack(A).std(0).mean())
print(f"PRODUCTION bonsai k={len(A)} (aa seeds, full-data): mean pair RMS {np.mean(r):.3f} LSB  min {min(r):.3f} max {max(r):.3f}  member-sd/px {sd:.3f}")
print("EVALSPLIT MEAN6 (starved, 6 recipes): mean pair RMS 9.478  member-sd/px 4.634")
print("EVALSPLIT SEED3 (starved, 1 recipe) : mean pair RMS 9.101  member-sd/px 3.676")
print(f"variance ratio prod/evalsplit6 = {(sd/4.634)**2:.3f}   prod/seed3 = {(sd/3.676)**2:.3f}")

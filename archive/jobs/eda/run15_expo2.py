import os, sys, re, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
def idx_of(n):
    m=re.search(r"_(\d{4})_V",n)
    if m: return int(m.group(1))
    m=re.search(r"frame_(\d+)",n); return int(m.group(1)) if m else None
for s in ["bonsai","chair","HCM0540","HCM0181"]:
    src=os.path.join(SET2,s) if s!="HCM0181" else os.path.join(PUB,s)
    d=os.path.join(src,"train/images"); ns=sorted(os.listdir(d))
    L=[];I=[]
    for n in ns:
        im=cv2.imread(os.path.join(d,n),cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        L.append(im.mean()); I.append(idx_of(n))
    L=np.array(L); I=np.array(I,float)
    o=np.argsort(I); L=L[o]; I=I[o]
    # how well can luminance be predicted by linear interpolation from immediate neighbours?
    pred=(L[:-2]+L[2:])/2
    interp_err=np.abs(pred-L[1:-1])
    print(f"{s:9s} n={len(L)} lum {L.min():.1f}..{L.max():.1f} range={L.max()-L.min():.1f} std={L.std():.2f}")
    print(f"          neighbour-interpolation abs err: med={np.median(interp_err):.2f} p90={np.percentile(interp_err,90):.2f} max={interp_err.max():.2f}")
    print(f"          => {100*(1-np.median(interp_err)/L.std()):.0f}% of the per-frame luminance variation is INTERPOLABLE from adjacent train frames")

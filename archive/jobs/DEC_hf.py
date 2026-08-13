"""Free discriminator: is the scale_reg gain BLUR, or fidelity? Gradient-RMS + radial band power vs GT."""
import os,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ES="/mnt/d/avv/evalsplit/bonsai"
D={"GT":None,"sr0":"/mnt/d/avv/r35_scalereg/sr0/eval_png","sr0.003":"/mnt/d/avv/r35_scalereg/sr0.003/eval_png",
   "K1(0.01)":"/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png","c42(0.01)":"/mnt/d/avv/r36_shape/sr001seed42/eval_png",
   "aniso":"/mnt/d/avv/r36_shape/aniso001/eval_png","sr0.03":"/mnt/d/avv/r35_scalereg/sr0.03/eval_png",
   "sr0.1":"/mnt/d/avv/r36_shape/sr01/eval_png"}
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
stems=sorted(gt)
def g(a):
    a=a.mean(2)
    return float(np.sqrt(((np.diff(a,axis=0)**2).mean()+(np.diff(a,axis=1)**2).mean())/2))
def bands(a):
    a=a.mean(2); F=np.abs(np.fft.rfft2(a-a.mean()))**2
    h,w=a.shape; fy=np.fft.fftfreq(h)[:,None]; fx=np.fft.rfftfreq(w)[None,:]
    r=np.sqrt(fy**2+fx**2)
    return [float(F[(r>=lo)&(r<hi)].mean()) for lo,hi in [(0.02,0.08),(0.08,0.2),(0.2,0.35),(0.35,0.51)]]
res={k:[] for k in D}; bnd={k:[] for k in D}
for st in stems:
    for k,v in D.items():
        a=np.asarray(Image.open(gt[st] if k=="GT" else f"{v}/{st}.png").convert("RGB"),dtype=np.float64)
        res[k].append(g(a)); bnd[k].append(bands(a))
print(f"{'dir':>10} {'gradRMS':>8} {'/GT':>6}   low     mid     high    vhigh   (band power ratio to GT)")
G=np.array(bnd["GT"]).mean(0)
for k in D:
    b=np.array(bnd[k]).mean(0)
    print(f"{k:>10} {np.mean(res[k]):8.4f} {np.mean(res[k])/np.mean(res['GT']):6.3f}   "+"  ".join(f"{x:6.3f}" for x in b/G),flush=True)

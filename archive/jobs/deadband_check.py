"""Verify the premise the deadband fix rests on: is png_ens EXACTLY round(mean of members)?
If yes, the float mean is recoverable with no new renders and the fix is exact."""
import os,sys,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
A=lambda p: np.asarray(Image.open(p).convert("RGB"),dtype=np.float64)/255.
CHAIR="/mnt/d/avv/r14/chair_aa42/test_png /mnt/d/avv/r14/chair_aa7/test_png /mnt/d/avv/r14/chair_aa13/test_png /mnt/d/avv/r17/chair_ema099_seed42/test_png /mnt/d/avv/r17/chair_ema099_seed7/test_png /mnt/d/avv/r17/chair_ema099_seed13/test_png /mnt/d/avv/r17/chair_depth_seed42/test_png /mnt/d/avv/r28_members/chair/test_png".split()
BONSAI="/mnt/d/avv/r14/bonsai_aa42/test_png /mnt/d/avv/r14/bonsai_aa7/test_png /mnt/d/avv/r14/bonsai_aa13/test_png /mnt/d/avv/r24_bonsai/aa101/test_png /mnt/d/avv/r24_bonsai/aa202/test_png /mnt/d/avv/r24_bonsai/aa303/test_png /mnt/d/avv/r28_members/bonsai/test_png".split()
def check(tag, ens_dir, dirs, w=None):
    stems=sorted(f[:-4] for f in os.listdir(ens_dir) if f.endswith(".png"))[:3]
    for s in stems:
        e=np.asarray(Image.open(f"{ens_dir}/{s}.png").convert("RGB")).astype(np.int64)
        ms=[A(f"{d}/{s}.png") for d in dirs]
        if w is None: m=np.mean(ms,axis=0)
        else:
            wa=np.array(w,dtype=np.float64); wa=wa/wa.sum()
            m=sum(wi*mi for wi,mi in zip(wa,ms))
        q=np.floor(m*255+0.5).astype(np.int64)
        print(f"  {tag:>10} {s:>16} exact={np.mean(q==e):.6f}  max|d|={np.abs(q-e).max()}  "
              f"subLSB_frac(|255*(m)-e|<0.5)={np.mean(np.abs(m*255-e)<0.5):.4f}")
print("VIDEO (uniform mean):")
check("chair k=8","/mnt/d/avv/r29/video_ens/chair/png_ens",CHAIR)
check("bonsai k=7","/mnt/d/avv/r29/video_ens/bonsai/png_ens",BONSAI)
print("\nTOWER (4:1:1 weighted over r22 png_ens, mip3d, r28 member):")
for T in ["HCM0421","HCM0539"]:
    d=[f"/mnt/d/avv/r22/tower_ens/{T}/png_ens",f"/mnt/d/avv/r25_mip3d/{T}/test_png",f"/mnt/d/avv/r28_members/{T}/test_png"]
    if all(os.path.isdir(x) for x in d): check(T,f"/mnt/d/avv/r29/tower_ens/{T}/png_ens",d,w=[4,1,1])
    else: print(f"  {T}: missing {[x for x in d if not os.path.isdir(x)]}")

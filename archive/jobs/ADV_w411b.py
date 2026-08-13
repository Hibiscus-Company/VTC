import numpy as np, os, glob
from PIL import Image
from scipy.ndimage import laplace
Image.MAX_IMAGE_PIXELS=None
for T in ["HCM0539","HCM0644"]:
    R22=f"/mnt/d/avv/r22/tower_ens/{T}/png_ens"; MIP=f"/mnt/d/avv/r25_mip3d/{T}/test_png"
    M28=f"/mnt/d/avv/r28_members/{T}/test_png"; A_D=f"/mnt/d/avv/r29/tower_ens/{T}/png_ens"
    stems=sorted(os.path.basename(p) for p in glob.glob(A_D+"/*.png"))[:4]
    L=lambda d,s: np.asarray(Image.open(os.path.join(d,s)).convert("RGB"),dtype=np.float32)
    hf=lambda x: float(np.sqrt((laplace(x.mean(2))**2).mean()))
    HA=HC=0; dev_e=dev_m1=dev_m2=0
    for s in stems:
        e,m1,m2=L(R22,s),L(MIP,s),L(M28,s)
        A=(4*e+m1+m2)/6.0; C=(6*e+m1+m2)/8.0     # C = TRUE uniform 8-way
        HA+=hf(A); HC+=hf(C)
        dev_e+=float(np.sqrt(((e-C)**2).mean())); dev_m1+=float(np.sqrt(((m1-C)**2).mean()))
        dev_m2+=float(np.sqrt(((m2-C)**2).mean()))
    n=len(stems)
    print(f"{T} n={n}  HF(4:1:1)={HA/n:.5f}  HF(6:1:1 uniform)={HC/n:.5f}  "
          f"dHF={100*(HC/n-HA/n)/(HA/n):+.3f}%")
    print(f"   deviation from the uniform-8 consensus:  6-member-mean e={dev_e/n:.3f}   "
          f"mip555={dev_m1/n:.3f}   r28mip={dev_m2/n:.3f}")
    # effective k
    for lbl,w in [("4:1:1",[4/6/6]*6+[1/6,1/6]),("6:1:1",[1/8]*8)]:
        w=np.array(w); print(f"   k_eff({lbl}) = {1/ (w**2).sum():.4f}")

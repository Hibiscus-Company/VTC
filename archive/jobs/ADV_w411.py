import numpy as np, os, glob
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
T=os.environ.get("T","HCM0539")
R22=f"/mnt/d/avv/r22/tower_ens/{T}/png_ens"
MIP=f"/mnt/d/avv/r25_mip3d/{T}/test_png"
M28=f"/mnt/d/avv/r28_members/{T}/test_png"
A_D=f"/mnt/d/avv/r29/tower_ens/{T}/png_ens"
B_D=f"/mnt/d/avv/r30/tower_ens/{T}/png_ens"
stems=sorted(os.path.basename(p) for p in glob.glob(A_D+"/*.png"))[:6]
def L(d,s): return np.asarray(Image.open(os.path.join(d,s)).convert("RGB"),dtype=np.float32)
rows=[]
for s in stems:
    e,m1,m2=L(R22,s),L(MIP,s),L(M28,s)
    A=L(A_D,s); B=L(B_D,s)
    # reproduce 4:1:1 exactly (float, no rounding) and build 6:1:1
    A_f=(4*e+1*m1+1*m2)/6.0
    C_f=(6*e+1*m1+1*m2)/8.0
    Ar=np.clip(A_f+0.5,0,255).astype(np.uint8).astype(np.float32)
    Cr=np.clip(C_f+0.5,0,255).astype(np.uint8).astype(np.float32)
    repro=np.abs(Ar-A).max()
    dCA=Cr-A; dBA=B-A
    rC=float(np.sqrt((dCA**2).mean())); rB=float(np.sqrt((dBA**2).mean()))
    cos=float((dCA*dBA).sum()/(np.linalg.norm(dCA)*np.linalg.norm(dBA)+1e-9))
    # member spread of the shipped 8-way pool proxy (ens dir vs mips)
    rows.append((s,repro,rC,rB,cos))
    print(f"{s[:34]} reproA_maxerr={repro:.0f}  RMS(C-A)={rC:.4f}  RMS(B-A)={rB:.4f}  ratio={rC/rB:.4f}  cos={cos:+.4f}")
a=np.array([[r[1],r[2],r[3],r[4]] for r in rows])
print(f"\n{T}  n={len(rows)}  reproA maxerr<= {a[:,0].max():.0f}")
print(f"MEAN RMS(C-A)={a[:,1].mean():.4f}  MEAN RMS(B-A)={a[:,2].mean():.4f}  ratio={a[:,1].mean()/a[:,2].mean():.4f}  mean cos={a[:,3].mean():+.4f}")

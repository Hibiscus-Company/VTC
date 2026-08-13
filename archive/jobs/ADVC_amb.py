"""Exact ambiguity accounting on the 28 eval holes: is the mean's fidelity gain fully explained
by member disagreement (variance reduction), or does a third variable (smoothing) contribute?"""
import os,itertools,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
B="/mnt/d/avv/bonsai_eval"; GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"
A6=["K1_noUT_aa","K4_pC_seed1k","K4_pC_seed7","eps10","eps20","ppisp_pc"]
gt_by={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
stems=sorted(f[:-4] for f in os.listdir(f"{B}/{A6[0]}/eval_png") if f.endswith(".png"))
mse_m=np.zeros(6); mse_e=0.; amb=0.; mse_b=0.; mse_pair=0.; ladder={k:0. for k in range(1,7)}
rng=np.random.default_rng(0)
for s in stems:
    A=np.stack([np.asarray(Image.open(f"{B}/{a}/eval_png/{s}.png").convert("RGB"),dtype=np.float64) for a in A6])
    g=np.asarray(Image.open(f"{GT}/{gt_by[s]}").convert("RGB"),dtype=np.float64)
    fb=A.mean(0)
    for i in range(6): mse_m[i]+=((A[i]-g)**2).mean()
    mse_e+=((fb-g)**2).mean()
    amb+=np.mean([((A[i]-fb)**2).mean() for i in range(6)])
    b=np.asarray(Image.open(f"/mnt/d/avv/ADVC/K1_blur05/{s}.png").convert("RGB"),dtype=np.float64)
    mse_b+=((b-g)**2).mean()
    mse_pair+=((A[[1,2]].mean(0)-g)**2).mean()
    for k in range(1,7):
        acc=[]
        for _ in range(20):
            idx=rng.choice(6,k,replace=False); acc.append(((A[idx].mean(0)-g)**2).mean())
        ladder[k]+=np.mean(acc)
n=len(stems); mse_m/=n; mse_e/=n; amb/=n; mse_b/=n; mse_pair/=n
P=lambda m: 10*np.log10(255.0**2/m)
print(f"n={n}  member MSE (LSB^2): {np.round(mse_m,2)}  mean {mse_m.mean():.2f}")
print(f"AMBIGUITY DECOMPOSITION  MSE(mean) = mean_i MSE_i - ambiguity")
print(f"   {mse_e:.3f}  =  {mse_m.mean():.3f} - {amb:.3f}  = {mse_m.mean()-amb:.3f}   (identity check)")
print(f"PSNR: best member(K1) {P(mse_m[0]):.4f}  avg member {P(mse_m.mean()):.4f}  MEAN6 {P(mse_e):.4f}")
print(f"SMOOTHING CONTROL K1+gauss0.5 (HF matched to mean6): PSNR {P(mse_b):.4f}  vs K1 {P(mse_m[0]):.4f}  d={P(mse_b)-P(mse_m[0]):+.4f} dB")
print(f"SEED-PAIR mean (K4_pC_seed7+seed1k): PSNR {P(mse_pair):.4f}  vs its members {P(mse_m[1]):.4f}/{P(mse_m[2]):.4f}")
print("k-ladder (random k-subsets, 20 draws/frame):")
for k in range(1,7):
    m=ladder[k]/n; print(f"   k={k}  MSE {m:7.3f}  PSNR {P(m):7.4f}  pred(1/k law) {mse_m.mean()-amb*6/5*(1-1.0/k):7.3f}")

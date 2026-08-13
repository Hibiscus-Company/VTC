"""R6: bias/variance of the ensemble.  How much of the residual can ANY amount of
   pixel-mean ensembling remove, and what is the irreducible floor?"""
import sys, json, numpy as np
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
POOL = [n for n in NAMES[:-1] if n != "sh3"]      # 20 distinct variants
pid = [IDX[n] for n in POOL]
H, W, N = 989, 1320, 60
VIEWS = list(range(0, 60, 3))
rng = np.random.RandomState(0)

# --- MSE-only depth curve over many random subsets (cheap, no LPIPS) ---
print("== MSE vs ensemble depth (20 distinct variants, 8 random subsets each) ==")
mse_k = {}
for k in [1, 2, 3, 4, 6, 8, 12, 16, 20]:
    acc = []
    for rep in range(8 if k < 20 else 1):
        sel = rng.choice(len(pid), k, replace=False) if k < 20 else np.arange(20)
        s = 0.0
        for i in VIEWS:
            m = np.mean([R[pid[j], i].astype(np.float32) for j in sel], 0) / 255.0
            g = G[i].astype(np.float32) / 255.0
            s += ((m - g) ** 2).mean()
        acc.append(s / len(VIEWS))
    mse_k[k] = float(np.mean(acc))
    print(f"  k={k:2d}  MSE {mse_k[k]:.6f}  PSNR {10*np.log10(1/mse_k[k]):7.4f}")

# fit MSE(k) = B + V/k
ks = np.array(sorted(mse_k)); ms = np.array([mse_k[k] for k in ks])
A = np.stack([np.ones_like(ks, float), 1.0 / ks], 1)
coef, *_ = np.linalg.lstsq(A, ms, rcond=None)
B, V = coef
print(f"\n  fit MSE(k) = B + V/k :  B(bias^2) {B:.6f}  V {V:.6f}")
print(f"  at k=4 : total {mse_k[4]:.6f}  bias-part {100*B/mse_k[4]:.1f}%  var-part {100*(V/4)/mse_k[4]:.1f}%")
print(f"  k=inf floor PSNR {10*np.log10(1/max(B,1e-9)):.4f} dB   (k=4 is {10*np.log10(1/mse_k[4]):.4f})")
print(f"  headroom from k=4 -> k=inf : {10*np.log10(mse_k[4]/max(B,1e-9)):.4f} dB PSNR")

# --- full composite for a few depths (with field) ---
print("\n== composite vs depth (with median lens field, 20 views) ==")
res = {}
for k in [1, 2, 4, 8, 20]:
    accs = []
    reps = 3 if k < 20 else 1
    for rep in range(reps):
        sel = rng.choice(len(pid), k, replace=False) if k < 20 else np.arange(20)
        P = S = L = 0.0
        for i in VIEWS:
            m = np.mean([R[pid[j], i].astype(np.float32) for j in sel], 0) / 255.0
            m = apply_field(np.clip(m, 0, 1), field)
            g = G[i].astype(np.float32) / 255.0
            a, b, c = score(m, g)
            P += a; S += b; L += c
        n = len(VIEWS)
        accs.append((P / n, S / n, L / n, comp(P / n, S / n, L / n)))
    a = np.mean(accs, 0)
    res[k] = a.tolist()
    print(f"  k={k:2d}  PSNR {a[0]:7.4f} SSIM {a[1]:.5f} LPIPS {a[2]:.5f}  comp {a[3]:8.4f}")

json.dump(dict(mse_k=mse_k, B=float(B), V=float(V), comp=res), open(OUT + "/r6.json", "w"))

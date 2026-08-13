"""R6b: bias/variance on a HOMOGENEOUS pool (15 same-quality variants)."""
import sys, json, numpy as np
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
POOL = ["m31b_nolpips", "sh3", "m31b_taillpips", "gsplatB10ut8M", "gsplatB12ut8Ms7",
        "e15ceil95", "gsplatB9ut", "gsplatB4warm", "gsplatB2", "e17visnorm",
        "gsplatB1", "gsplatB8pure", "gsplatB3", "gsplatB7ppisp2", "gsplatB5affine"]
pid = [IDX[n] for n in POOL]
VIEWS = list(range(0, 60, 3))
rng = np.random.RandomState(1)

print("== homogeneous pool of %d, MSE vs depth (12 subsets each) ==" % len(POOL))
mse_k = {}
for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
    acc = []
    reps = 12 if k < 15 else 1
    for rep in range(reps):
        sel = rng.choice(len(pid), k, replace=False) if k < 15 else np.arange(15)
        s = 0.0
        for i in VIEWS:
            m = np.mean([R[pid[j], i].astype(np.float32) for j in sel], 0) / 255.0
            s += ((m - G[i].astype(np.float32) / 255.0) ** 2).mean()
        acc.append(s / len(VIEWS))
    mse_k[k] = float(np.mean(acc))
    print(f"  k={k:2d}  MSE {mse_k[k]:.6f}  PSNR {10*np.log10(1/mse_k[k]):7.4f}")

ks = np.array(sorted(mse_k)); ms = np.array([mse_k[k] for k in ks])
A = np.stack([np.ones_like(ks, float), 1.0 / ks], 1)
(B, V), *_ = np.linalg.lstsq(A, ms, rcond=None)
print(f"\n  MSE(k)=B+V/k :  B {B:.6f}  V {V:.6f}   pred k=inf PSNR {10*np.log10(1/B):.4f}")
for k in [4, 6, 7, 15]:
    print(f"   k={k:2d}: bias share {100*B/(B+V/k):.1f}%   PSNR {10*np.log10(1/(B+V/k)):.4f}")
print(f"  headroom k=7 -> k=inf : {10*np.log10((B+V/7)/B):.4f} dB")

print("\n== composite vs depth (field applied) ==")
comp_k = {}
for k in [1, 2, 4, 6, 8, 11, 15]:
    accs = []
    reps = 4 if k < 15 else 1
    for rep in range(reps):
        sel = rng.choice(len(pid), k, replace=False) if k < 15 else np.arange(15)
        P = S = L = 0.0
        for i in VIEWS:
            m = np.mean([R[pid[j], i].astype(np.float32) for j in sel], 0) / 255.0
            m = apply_field(np.clip(m, 0, 1), field)
            g = G[i].astype(np.float32) / 255.0
            a, b, c = score(m, g)
            P += a; S += b; L += c
        n = len(VIEWS)
        accs.append((P / n, S / n, L / n, comp(P / n, S / n, L / n)))
    a = np.mean(accs, 0); comp_k[k] = a.tolist()
    print(f"  k={k:2d}  PSNR {a[0]:7.4f} SSIM {a[1]:.5f} LPIPS {a[2]:.5f}  comp {a[3]:8.4f}")
json.dump(dict(mse_k=mse_k, B=float(B), V=float(V), comp_k=comp_k), open(OUT + "/r6b.json", "w"))

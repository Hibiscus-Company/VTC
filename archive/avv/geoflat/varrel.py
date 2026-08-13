#!/usr/bin/env python
"""How exactly does our fine texture fall short of GT's, as a function of how textured we already
are? Bin flat pixels by our own local HF std and read off GT's. A pure ratio => the deficit is
multiplicative (a gain fixes it, and gain is already at its optimum). An intercept => GT has a
texture FLOOR we lack entirely, which a gain can never supply."""
import os, sys
import numpy as np, cv2
cv2.setNumThreads(8)
CD = "/mnt/d/avv/geoflat/cache_HCM0181"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 2.0)
lstd = lambda x, w=9: np.sqrt(np.maximum(cv2.blur(x * x, (w, w)) - cv2.blur(x, (w, w)) ** 2, 0))

stems = sorted(f[:-4] for f in os.listdir(CD) if f.endswith(".npz"))[:N]
EDG = np.linspace(0, 24, 13)          # our local HF std, in 1/255
S = np.zeros((12, 4))                 # n, sum gt_std, sum our_std, sum cov
rho_band = {k: [] for k in ("L0", "L1", "L2", "hp2")}
gain_band = {k: [] for k in ("L0", "L1", "L2", "hp2")}
for s in stems:
    z = np.load(os.path.join(CD, s + ".npz"))
    B = z["B"].astype(np.float32); G = z["G"].astype(np.float32); D = z["D"].astype(np.float32)
    flat = D > 6
    by = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY); gy = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
    # --- per-band coherence and Wiener gain inside flat
    bands = {}
    b0, g0 = by, gy
    for i, sg in enumerate((1.0, 2.0, 4.0)):
        bb, gg = cv2.GaussianBlur(by, (0, 0), sg), cv2.GaussianBlur(gy, (0, 0), sg)
        bands[f"L{i}"] = (b0 - bb, g0 - gg); b0, g0 = bb, gg
    bands["hp2"] = (hp(by), hp(gy))
    for k, (x, y) in bands.items():
        a, b = x[flat], y[flat]
        a = a - a.mean(); b = b - b.mean()
        r = float((a * b).mean() / (a.std() * b.std() + 1e-12))
        rho_band[k].append(r)
        gain_band[k].append(float((a * b).mean() / ((a * a).mean() + 1e-12)))
    # --- variance relation
    sb, sg_ = lstd(hp(by)) * 255, lstd(hp(gy)) * 255
    idx = np.clip(np.digitize(sb[flat], EDG) - 1, 0, 11)
    v_b, v_g = sb[flat], sg_[flat]
    np.add.at(S, (idx, 0), 1.0); np.add.at(S, (idx, 1), v_g)
    np.add.at(S, (idx, 2), v_b); np.add.at(S, (idx, 3), v_g ** 2)

print("flat-region per-band coherence with GT (n=%d imgs)" % len(stems))
print(f"{'band':>5} {'rho':>7} {'wiener g*':>10}")
for k in rho_band:
    print(f"{k:>5} {np.mean(rho_band[k]):7.4f} {np.mean(gain_band[k]):10.4f}")
print("\nlocal HF std (1/255, 9x9 window), our value -> GT value, flat pixels only")
print(f"{'our bin':>12} {'n(M)':>7} {'our':>7} {'GT':>7} {'ratio':>7}")
for i in range(12):
    if S[i, 0] < 1e4:
        continue
    n, sg_, sb = S[i, 0], S[i, 1] / S[i, 0], S[i, 2] / S[i, 0]
    print(f"{EDG[i]:5.1f}-{EDG[i+1]:5.1f} {n/1e6:7.2f} {sb:7.3f} {sg_:7.3f} {sg_/sb:7.3f}")
w = S[:, 0] > 1e4
x = S[w, 2] / S[w, 0]; y = S[w, 1] / S[w, 0]; n = S[w, 0]
A = np.stack([x, np.ones_like(x)], 1)
c = np.linalg.lstsq(A * np.sqrt(n)[:, None], y * np.sqrt(n), rcond=None)[0]
print(f"\nweighted fit  GT_std = {c[0]:.4f} * our_std + {c[1]:.4f}   (1/255)")
print(f"  -> implied additive texture floor we lack: sqrt(max(GT^2-(a*our)^2)) at our_std=2: "
      f"{np.sqrt(max((c[0]*2+c[1])**2-(c[0]*2)**2,0)):.3f}/255")

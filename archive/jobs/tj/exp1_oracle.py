"""EXP-1  Oracle upper bounds on the PHOTOMETRIC-CORRECTION axis (production harness).

Baseline = float pixel-mean of the 4 UT members (= the k4 production ensemble).
Every "oracle" below is FIT ON THE TEST GT ITSELF -- it is not shippable, it exists only to
BOUND how much any honest estimator on this axis could ever deliver.  If the oracle is small
the whole axis is dead and we stop; if it is large we then build the honest train-fit version.

O1  per-view per-channel gain+bias                (6 params / view)
O2  per-view 3x3 colour matrix + bias             (12 params / view)
O3  per-view smooth spatial gain+bias field, deg-2 poly per channel (36 params / view)
O4  ONE global per-pixel additive bias field, shared by all 60 views (H*W*3 params)
O5  O4 + O1 stacked (field then per-view exposure)
"""
import numpy as np, hlib, json

hlib.init()
N = hlib.names(); R = hlib.renders(); G = np.asarray(hlib.gt()).astype(np.float32) / 255.0
MEM = ['gsplatB9ut', 'gsplatB10ut8M', 'gsplatB11ut60k', 'gsplatB12ut8Ms7']
idx = [N.index(m) for m in MEM]

base = np.zeros(G.shape, np.float32)
for i in idx:
    base += R[i].astype(np.float32) / 255.0
base /= len(idx)
n, H, W, _ = base.shape
print(f"base {base.shape}", flush=True)

res = {}
res['base'] = hlib.score(base, np.asarray(hlib.gt()), "BASE k4 float-mean")

# ---------------- O1 per-view per-channel gain+bias ----------------
o1 = np.empty_like(base)
for i in range(n):
    for c in range(3):
        x = base[i, :, :, c].ravel().astype(np.float64)
        y = G[i, :, :, c].ravel().astype(np.float64)
        A = np.stack([x, np.ones_like(x)], 1)
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        o1[i, :, :, c] = a * base[i, :, :, c] + b
res['O1'] = hlib.score(o1, np.asarray(hlib.gt()), "O1 per-view gain+bias (6p)")

# ---------------- O2 per-view 3x3 CCM + bias ----------------
o2 = np.empty_like(base)
for i in range(n):
    X = base[i].reshape(-1, 3).astype(np.float64)
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = G[i].reshape(-1, 3).astype(np.float64)
    Msol = np.linalg.lstsq(A, Y, rcond=None)[0]
    o2[i] = (A @ Msol).reshape(H, W, 3)
res['O2'] = hlib.score(o2, np.asarray(hlib.gt()), "O2 per-view CCM+bias (12p)")

# ---------------- O3 per-view smooth spatial gain+bias (deg-2 poly) ----------------
yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
POLY = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1).astype(np.float64)
POLY_f = POLY.reshape(-1, 6)
o3 = np.empty_like(base)
for i in range(n):
    for c in range(3):
        r = base[i, :, :, c].reshape(-1, 1).astype(np.float64)
        A = np.concatenate([POLY_f * r, POLY_f], 1)          # 12 params
        y = G[i, :, :, c].ravel().astype(np.float64)
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        o3[i, :, :, c] = (A @ w).reshape(H, W)
res['O3'] = hlib.score(o3, np.asarray(hlib.gt()), "O3 per-view smooth field (36p)")

# ---------------- O4 one global per-pixel bias field (oracle, fit on TEST) ----------------
bias = (G - base).mean(0)                                    # H,W,3
o4 = base + bias[None]
res['O4'] = hlib.score(o4, np.asarray(hlib.gt()), "O4 global per-px bias (oracle)")
np.save('/home/bkai/.claude/jobs/1c9cf7e9/tmp/tj/bias_testoracle.npy', bias)
print("   bias field: mean %.5f  std %.5f  absmax %.5f" % (bias.mean(), bias.std(), np.abs(bias).max()))

# ---------------- O5 O4 then per-view gain+bias ----------------
o5 = np.empty_like(base)
for i in range(n):
    for c in range(3):
        x = o4[i, :, :, c].ravel().astype(np.float64)
        y = G[i, :, :, c].ravel().astype(np.float64)
        A = np.stack([x, np.ones_like(x)], 1)
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        o5[i, :, :, c] = a * o4[i, :, :, c] + b
res['O5'] = hlib.score(o5, np.asarray(hlib.gt()), "O5 = O4 + O1")

b = res['base']['score']
print("\n%-34s %8s %8s" % ("config", "score", "dScore"))
for k, v in res.items():
    print("%-34s %8.4f %+8.4f" % (v['tag'], v['score'], v['score'] - b))
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'pv'} for k, v in res.items()},
          open('/home/bkai/.claude/jobs/1c9cf7e9/tmp/tj/exp1.json', 'w'), indent=1)

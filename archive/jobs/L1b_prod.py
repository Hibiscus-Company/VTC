"""ROUND 2: refine ALIGN-THEN-MERGE and test it in the FULL PRODUCTION CHAIN.
prod chain = ensemble -> median lens field -> JPEG q100 ss2 prog optimize.
Everything is measured after the exact shipped encode, so no proxy step is left unverified."""
import os, sys, json, time, io
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
import agg_lib as A

POOL = sys.argv[1]; MODE = sys.argv[2]      # MODE: refine | prod
POOLS = {
 "p7": ["gsplatB9ut","gsplatB10ut8M","gsplatB11ut60k","gsplatB12ut8Ms7","m31b_taillpips","e17visnorm","gsplatB8pure"],
 "p4": ["gsplatB9ut","gsplatB10ut8M","gsplatB11ut60k","gsplatB12ut8Ms7"],
 "pB": ["gsplatB1","gsplatB2","gsplatB3","gsplatB4warm"],
 "pC": ["e15ceil95","e16app","e17visnorm","m31b_taillpips"],
 "pD": ["gsplatB5affine","gsplatB7ppisp2","gsplatB8pure","m31b_nolpips"],
}
V = POOLS[POOL]
stems, gt_by = A.stems_for(V)
dev = A.init("cuda")
H, W = 989, 1320
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
FLD = np.load("/mnt/d/avv/fields/HCM0181.npy")
ENC = dict(format="JPEG", quality=100, subsampling=2, optimize=True, progressive=True)
print(f"pool={POOL} mode={MODE} k={len(V)} views={len(stems)}", flush=True)

def mkdis():
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    d.setFinestScale(0); d.setPatchSize(8); d.setPatchStride(3)
    d.setUseMeanNormalization(True); d.setUseSpatialPropagation(True)
    d.setVariationalRefinementIterations(5)
    return d
DIS = mkdis()
g8 = lambda x: cv2.cvtColor((np.clip(x,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

def align_merge(X, ref, clamp=1.0, smooth=0.0, interp=cv2.INTER_LANCZOS4):
    gm = g8(ref); acc = np.zeros_like(ref)
    for i in range(X.shape[0]):
        F = DIS.calc(gm, g8(X[i]), None)
        if smooth > 0: F = cv2.GaussianBlur(F, (0,0), smooth)
        m = np.sqrt((F**2).sum(-1, keepdims=True))
        F = F * np.minimum(1.0, clamp/np.maximum(m, 1e-6))
        acc += cv2.remap(X[i], gx+F[...,0], gy+F[...,1], interp, borderMode=cv2.BORDER_REFLECT)
    return acc / X.shape[0]

def field(img):
    fu = cv2.resize(FLD, (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.remap(img.astype(np.float32), (gx+fu[...,0]).astype(np.float32),
                     (gy+fu[...,1]).astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

def jpeg(img):
    b = io.BytesIO()
    Image.fromarray(np.clip(np.round(img*255),0,255).astype(np.uint8)).save(b, **ENC)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), np.float32)/255.

sc = A.Scorer(dev); t0 = time.time()
for n, s in enumerate(stems):
    X = np.stack([A.load(A.mdir(v), s) for v in V]).astype(np.float32)
    M = X.mean(0); gt = A.gt_tensor(gt_by[s], dev)
    if MODE == "refine":
        A0 = align_merge(X, M, 1.0, 0.0)
        cand = {"mean": M, "al_c1.0_s0": A0,
                "al_c0.5_s0": align_merge(X, M, 0.5, 0.0),
                "al_c2.0_s0": align_merge(X, M, 2.0, 0.0),
                "al_c1.0_s2": align_merge(X, M, 1.0, 2.0),
                "al_c1.0_s5": align_merge(X, M, 1.0, 5.0),
                "al_iter2":   align_merge(X, A0, 1.0, 0.0),
                "al_extrap1.25": M + 1.25*(A0-M),
                "al_extrap1.50": M + 1.50*(A0-M)}
    elif MODE == "extrap":
        A0 = align_merge(X, M, 1.0, 0.0)
        cand = {"mean": M, "al_l1.00": A0,
                "al_extrap1.25": M + 1.25*(A0-M),
                "al_extrap1.50": M + 1.50*(A0-M),
                "al_extrap2.00": M + 2.00*(A0-M)}
    else:
        Aq = align_merge(X, M, 1.0, 0.0)
        cand = {"mean":               M,
                "align":              Aq,
                "mean+field":         field(M),
                "align+field":        field(Aq),
                "mean+field+jpeg":    jpeg(field(M)),
                "align+field+jpeg":   jpeg(field(Aq)),
                "mean+jpeg":          jpeg(M),
                "align+jpeg":         jpeg(Aq)}
    for k, im in cand.items(): sc.add(k, im, gt)
    if (n+1) % 15 == 0: print(f"  {n+1}/{len(stems)} ({time.time()-t0:.0f}s)", flush=True)

rows = sc.table()
base = dict((r[0], r[1]) for r in rows)["mean"]
print(f"\n{'variant':22s} {'SCORE':>9} {'d(mean)':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7}")
for k, s_, P, S, L, n_ in sorted(rows, key=lambda r: -r[1]):
    print(f"{k:22s} {s_:9.4f} {s_-base:+9.4f} {P:8.4f} {S:7.4f} {L:7.4f}")
json.dump({k: v for k, v in sc.acc.items()},
          open(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/L1b_{POOL}_{MODE}.json","w"))

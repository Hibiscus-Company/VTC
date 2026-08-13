"""PAIRED SIGNIFICANCE of align-then-merge vs pixel-mean, per view.
The project composite is LINEAR in mean(PSNR), mean(SSIM), mean(LPIPS), so the per-view
composite delta averages exactly to the reported delta -> a paired bootstrap over views is exact."""
import os, sys, json, time
import numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
import agg_lib as A

POOL = sys.argv[1]
POOLS = {
 "p7": ["gsplatB9ut","gsplatB10ut8M","gsplatB11ut60k","gsplatB12ut8Ms7","m31b_taillpips","e17visnorm","gsplatB8pure"],
 "pB": ["gsplatB1","gsplatB2","gsplatB3","gsplatB4warm"],
 "pC": ["e15ceil95","e16app","e17visnorm","m31b_taillpips"],
 "pD": ["gsplatB5affine","gsplatB7ppisp2","gsplatB8pure","m31b_nolpips"],
}
V = POOLS[POOL]; stems, gt_by = A.stems_for(V); dev = A.init("cuda")
H, W = 989, 1320
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
D = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
D.setFinestScale(0); D.setPatchSize(8); D.setPatchStride(3)
D.setUseMeanNormalization(True); D.setUseSpatialPropagation(True); D.setVariationalRefinementIterations(5)
g8 = lambda x: cv2.cvtColor((np.clip(x,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

def am(X, ref):
    gm = g8(ref); acc = np.zeros_like(ref)
    for i in range(X.shape[0]):
        F = D.calc(gm, g8(X[i]), None)
        m = np.sqrt((F**2).sum(-1, keepdims=True)); F = F*np.minimum(1.0, 1.0/np.maximum(m,1e-6))
        acc += cv2.remap(X[i], gx+F[...,0], gy+F[...,1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    return acc/X.shape[0]

comp = lambda P,S,L: 100*(0.4*(1-L)+0.3*S+0.3*min(P/50.,1.))
rec = []
for n, s in enumerate(stems):
    X = np.stack([A.load(A.mdir(v), s) for v in V]).astype(np.float32)
    M = X.mean(0); gt = A.gt_tensor(gt_by[s], dev)
    a = A.metrics(M, gt, dev); b = A.metrics(am(X, M), gt, dev)
    rec.append([comp(*a), comp(*b), a[0], b[0], a[1], b[1], a[2], b[2]])
    if (n+1) % 20 == 0: print(f"  {n+1}/{len(stems)}", flush=True)
R = np.array(rec); d = R[:,1]-R[:,0]
n = len(d); se = d.std(ddof=1)/np.sqrt(n)
rng = np.random.default_rng(0)
bs = np.array([d[rng.integers(0,n,n)].mean() for _ in range(20000)])
print(f"\n=== PAIRED, pool={POOL}, n={n} views ===")
print(f"  mean dScore   = {d.mean():+.4f}   sd {d.std(ddof=1):.4f}  SE {se:.4f}  t = {d.mean()/se:.2f}")
print(f"  95% CI (bootstrap over views) = [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
print(f"  views improved = {(d>0).sum()}/{n}  ({100*(d>0).mean():.0f}%)   worst {d.min():+.4f}  best {d.max():+.4f}")
print(f"  dPSNR {np.mean(R[:,3]-R[:,2]):+.4f} dB   dSSIM {np.mean(R[:,5]-R[:,4]):+.6f}   dLPIPS {np.mean(R[:,7]-R[:,6]):+.6f}")
np.save(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/L1c_{POOL}.npy", R)

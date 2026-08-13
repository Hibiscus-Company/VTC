"""How much of the residual displacement is VIEW-CONSISTENT (a fittable field) vs VIEW-SPECIFIC?
CPU, MSE-only, HCM0181, 60 real test views, starting from the UNWARPED k4 ensemble.
LOO = fit the pooled field on 59 views, apply to the held-out one (honest)."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(6)
SCENE = "HCM0181"
st = stems(); gd, gm = gt_map(SCENE)
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
G8s = []; K8s = []; FL = []
for s in st:
    G8 = mlib.load_u8(os.path.join(gd, gm[s])); K8 = mlib.load_u8(os.path.join(K4, s + ".png"))
    G8s.append(G8); K8s.append(K8)
    FL.append(np.clip(dis.calc(cv2.cvtColor(G8, cv2.COLOR_RGB2GRAY),
                               cv2.cvtColor(K8, cv2.COLOR_RGB2GRAY), None), -6, 6))
FL = np.stack(FL)
print("per-view oracle flow rms  %.4f px" % np.sqrt((FL ** 2).sum(-1)).mean())
med_all = np.median(FL, 0)
print("view-median flow rms      %.4f px  (view-consistent part)" % np.sqrt((med_all ** 2).sum(-1)).mean())
print("per-view deviation rms    %.4f px  (view-specific part)"
      % np.sqrt(((FL - med_all) ** 2).sum(-1)).mean())
print("energy share view-consistent: %.1f%%"
      % (100 * (med_all ** 2).sum() * len(FL) / (FL ** 2).sum()))

H, W = FL.shape[1], FL.shape[2]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def warp(img8, f):
    return np.clip(cv2.remap(img8.astype(np.float32), (xx + f[..., 0]).astype(np.float32),
                             (yy + f[..., 1]).astype(np.float32), cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT), 0, 255).astype(np.uint8)
def up(f, ds):
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC) if ds > 1 else f
DSS = [16, 8, 4, 2, 1]
LOW = {ds: np.stack([cv2.resize(f, (W // ds, H // ds), interpolation=cv2.INTER_AREA) for f in FL])
       if ds > 1 else FL for ds in DSS}
shipped = np.load(f"/mnt/d/avv/fields/{SCENE}.npy")
res = {"raw": 0.0, "shipped_field": 0.0, "oracle_perview": 0.0}
for ds in DSS: res[f"LOOmed_ds{ds}"] = 0.0
n = 0
for i in range(len(st)):
    G = G8s[i].astype(np.float32) / 255.
    def mse(a): return ((a.astype(np.float32) / 255. - G) ** 2).mean()
    res["raw"] += mse(K8s[i]); res["shipped_field"] += mse(warp(K8s[i], up(shipped, 8)))
    res["oracle_perview"] += mse(warp(K8s[i], FL[i]))
    for ds in DSS:
        A = LOW[ds]
        loo = np.median(np.delete(A, i, axis=0), 0)
        res[f"LOOmed_ds{ds}"] += mse(warp(K8s[i], up(loo, ds)))
    n += 1
print("\n  variant             MSE       PSNR    dPSNR   dScore(PSNRterm)  %of oracle gain")
base = res["raw"] / n; orc = res["oracle_perview"] / n
p0 = 10 * np.log10(1 / base); po = 10 * np.log10(1 / orc)
for k, v in res.items():
    m = v / n; p = 10 * np.log10(1 / m)
    print(f"  {k:18s} {m:.6f} {p:8.4f} {p-p0:+8.4f} {0.6*(p-p0):+10.4f}      "
          f"{100*(p-p0)/(po-p0) if po != p0 else 0:7.1f}%")

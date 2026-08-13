#!/usr/bin/env python
"""Can we get COHERENT fine texture into flat regions?

The oracle says +1.60 of the +1.85 flat-region ceiling lives in the sigma-2 highpass band, and
that band is the one our renders lose. Our own HF there is incoherent with GT. But the TRAIN
PHOTOS contain the real physical texture of the same asphalt/roofs. Idea: render->render dense
flow (photometrically clean, same model, same exposure) transports a train PHOTO into the test
frame; take only its HF, only in flat regions.

This script measures COHERENCE only (no scoring): corr(HP(x), HP(GT)) inside the flat mask.
  own      our shipped render               <- the number to beat
  ph_w     nearest train photo, flow-warped <- the operator's raw material
  rn_w     nearest train render, flow-warped<- control: flow alignment without real texture
  ph_ctl   a RANDOM other train photo, warped with the SAME flow <- control: texture statistics
                                                                   without correspondence
  ph_nw    nearest train photo, NOT warped  <- control: value of the flow
"""
import os, sys, time
import numpy as np, cv2
from PIL import Image
cv2.setNumThreads(8); Image.MAX_IMAGE_PIXELS = None
SC = "HCM0181"
CD = f"/mnt/d/avv/geoflat/cache_{SC}"
TRD = f"/mnt/d/avv/output/{SC}_gsplatB9ut/train_renders"
TPD = f"/mnt/d/avv/data/phase1/public_set/{SC}/train/images"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

tstems = sorted(os.path.splitext(f)[0] for f in os.listdir(TRD))
tp = {os.path.splitext(f)[0]: f for f in os.listdir(TPD)}
tstems = [s for s in tstems if s in tp]
print(f"{len(tstems)} train views with both render and photo", flush=True)

# hoisted: thumbnail of every train render, once, for nearest-view lookup
TH = {}
TRG = {}
for s in tstems:
    a = cv2.imread(os.path.join(TRD, s + ".png"), cv2.IMREAD_GRAYSCALE)
    TRG[s] = a
    t = cv2.resize(a, (a.shape[1] // 8, a.shape[0] // 8)).astype(np.float32)
    TH[s] = (t - t.mean()) / (t.std() + 1e-6)

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
dis.setUseSpatialPropagation(True)
hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 2.0)


def corr(a, b, m):
    a = a[m]; b = b[m]
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-12))


stems = sorted(f[:-4] for f in os.listdir(CD) if f.endswith(".npz"))[:N]
rows = {k: [] for k in ("own", "ph_w", "rn_w", "ph_ctl", "ph_nw", "amp_own", "amp_ph", "amp_gt", "flowpx")}
rng = np.random.default_rng(0)
for i, s in enumerate(stems):
    z = np.load(os.path.join(CD, s + ".npz"))
    B = z["B"].astype(np.float32); G = z["G"].astype(np.float32); D = z["D"].astype(np.float32)
    flat = D > 6
    by = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
    t = cv2.resize(by, (by.shape[1] // 8, by.shape[0] // 8))
    t = (t - t.mean()) / (t.std() + 1e-6)
    sim = {k: float((t * v).mean()) for k, v in TH.items()}
    best = max(sim, key=sim.get)
    other = tstems[int(rng.integers(len(tstems)))]
    t0 = time.time()
    pv = (np.clip(by, 0, 1) * 255).astype(np.uint8)
    fl = dis.calc(pv, TRG[best], None)
    H, W = pv.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = (xx + fl[..., 0]).astype(np.float32), (yy + fl[..., 1]).astype(np.float32)

    def rd(p):
        return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.

    ph = rd(os.path.join(TPD, tp[best]))
    ph_w = cv2.remap(ph, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    rn_w = cv2.remap(cv2.imread(os.path.join(TRD, best + ".png"))[..., ::-1].astype(np.float32) / 255.,
                     mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    ph_o = cv2.remap(rd(os.path.join(TPD, tp[other])), mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    hg = hp(cv2.cvtColor(G, cv2.COLOR_RGB2GRAY))
    rows["own"].append(corr(hp(by), hg, flat))
    rows["ph_w"].append(corr(hp(cv2.cvtColor(ph_w, cv2.COLOR_RGB2GRAY)), hg, flat))
    rows["rn_w"].append(corr(hp(cv2.cvtColor(rn_w, cv2.COLOR_RGB2GRAY)), hg, flat))
    rows["ph_ctl"].append(corr(hp(cv2.cvtColor(ph_o, cv2.COLOR_RGB2GRAY)), hg, flat))
    rows["ph_nw"].append(corr(hp(cv2.cvtColor(ph, cv2.COLOR_RGB2GRAY)), hg, flat))
    rows["amp_own"].append(float(hp(by)[flat].std() * 255))
    rows["amp_ph"].append(float(hp(cv2.cvtColor(ph_w, cv2.COLOR_RGB2GRAY))[flat].std() * 255))
    rows["amp_gt"].append(float(hg[flat].std() * 255))
    rows["flowpx"].append(float(np.hypot(fl[..., 0], fl[..., 1])[flat].mean()))
    print(f"  {i} best={best} sim={sim[best]:.3f} flow={rows['flowpx'][-1]:.1f}px "
          f"own={rows['own'][-1]:.3f} ph_w={rows['ph_w'][-1]:.3f} rn_w={rows['rn_w'][-1]:.3f} "
          f"ctl={rows['ph_ctl'][-1]:.3f} ({time.time()-t0:.1f}s)", flush=True)

print("\nflat-region corr with GT high-pass (sigma 2), n =", len(stems))
for k, v in rows.items():
    print(f"{k:>8} {np.mean(v):8.4f}")

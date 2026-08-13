"""LATERAL CHROMATIC ABERRATION: does the render->photo displacement field differ per COLOUR CHANNEL?

The shipped lens field is fit on GRAYSCALE DIS flow and applied identically to R, G and B.
Real lenses have lateral CA: the red and blue images are radially scaled by different amounts
(typ. 0.2-1 px at the frame corner on a consumer drone lens).  COLMAP's SIMPLE_RADIAL is a single
achromatic k1 and gsplat pins distortion, so an LCA residual would land in exactly the same
bucket the achromatic field lives in -- and would be invisible to a grayscale flow fit, because
grayscale averages R and B and the LCA term cancels to first order.

PER-IMAGE, geometric, adds no high-frequency energy (it is a resample, like the shipped field).
This probe only asks whether the effect EXISTS above the fit noise; scoring comes after.
"""
import os, sys, time
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, os.path.join(HERE, "lens"))
from fieldlib import LooPool, gauss_smooth

DS = 8
TAGS = ["HCM0181", "HCM0193", "HCM0204"]
for TAG in TAGS:
    R = f"/mnt/d/avv/output/{TAG}_gsplatB9ut/test_poses_renders_png"
    G = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(G)}
    stems = sorted(s for s in gt_by if os.path.exists(os.path.join(R, s + ".png")))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = f"{HERE}/lens/cache/rgb_{TAG}.npz"
    if os.path.exists(out):
        z = np.load(out); st = {k: z[k] for k in ("R", "G", "B", "Y")}
    else:
        st = {k: [] for k in "RGBY"}
        t0 = time.time()
        for s in stems:
            r = np.asarray(Image.open(os.path.join(R, s + ".png")).convert("RGB"))
            g = np.asarray(Image.open(os.path.join(G, gt_by[s])).convert("RGB"))
            if r.shape != g.shape:
                continue
            H, W, _ = r.shape
            for i, c in enumerate("RGB"):
                fl = np.clip(dis.calc(g[..., i].copy(), r[..., i].copy(), None), -6, 6)
                st[c].append(cv2.resize(fl, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
            ry = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY); gy = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
            fl = np.clip(dis.calc(gy, ry, None), -6, 6)
            st["Y"].append(cv2.resize(fl, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
        st = {k: np.stack(v) for k, v in st.items()}
        np.savez_compressed(out, **st)
        print(f"{TAG}: {len(st['Y'])} pairs, {time.time()-t0:.0f}s", flush=True)

    med = {k: gauss_smooth(np.median(v, 0), 1) for k, v in st.items()}
    n, h, w, _ = st["Y"].shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    rx, ry = xx - cx, yy - cy
    rad = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
    ur, vr = rx / rad, ry / rad          # unit radial vector

    # per-view SEM of the R-B difference, so "exists" means "bigger than the fit noise"
    diff_stack = st["R"] - st["B"]
    sem = diff_stack.std(0, ddof=1) / np.sqrt(n)
    d = med["R"] - med["B"]
    dm = np.linalg.norm(d, axis=2)
    radial = d[..., 0] * ur + d[..., 1] * vr      # signed radial component of R-B
    # LCA signature: radial component grows ~linearly with radius
    rr = rad.ravel() / rad.max()
    cc = np.corrcoef(rr, radial.ravel())[0, 1]
    slope = np.polyfit(rr, radial.ravel(), 1)[0]
    print(f"\n== {TAG}  n={n}  field {w}x{h} (ds{DS})")
    print(f"   |Y field|          mean {np.linalg.norm(med['Y'],axis=2).mean():.4f} px   "
          f"max {np.linalg.norm(med['Y'],axis=2).max():.4f} px")
    print(f"   |R-B|              mean {dm.mean():.4f} px   max {dm.max():.4f} px   "
          f"p95 {np.percentile(dm,95):.4f}")
    print(f"   fit noise SEM|R-B| mean {np.linalg.norm(sem,axis=2).mean():.4f} px    "
          f"=> SNR {dm.mean()/max(np.linalg.norm(sem,axis=2).mean(),1e-9):.2f}")
    print(f"   radial(R-B) vs radius: corr {cc:+.3f}  slope {slope:+.4f} px/half-diag  "
          f"(pure LCA would be corr ~+-1)")
    print(f"   |R-Y| mean {np.linalg.norm(med['R']-med['Y'],axis=2).mean():.4f}  "
          f"|B-Y| mean {np.linalg.norm(med['B']-med['Y'],axis=2).mean():.4f}  "
          f"|G-Y| mean {np.linalg.norm(med['G']-med['Y'],axis=2).mean():.4f} px")
    sys.stdout.flush()

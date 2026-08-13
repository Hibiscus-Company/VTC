"""Residual decomposition of the production ensemble (k4 + train-fitted lens field) vs REAL test GT.
(a) radial power spectrum / coherence   (b) error concentration + spatial masks
(c) ensemble-variance vs error correlation
"""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(8)

SCENE = "HCM0181"
FLD = np.load(f"/mnt/d/avv/fields/{SCENE}.npy")
st = stems()
gd, gm = gt_map(SCENE)
H, W = 989, 1320

# ---------- radial bin map ----------
fy = np.fft.fftfreq(H)[:, None]
fx = np.fft.fftfreq(W)[None, :]
rad = np.sqrt(fy ** 2 + fx ** 2)
NB = 48
edges = np.linspace(0, 0.5, NB + 1)
bidx = np.clip(np.digitize(rad.ravel(), edges) - 1, 0, NB - 1)
cnt = np.bincount(bidx, minlength=NB).astype(np.float64)
wy = np.hanning(H)[:, None]; wx = np.hanning(W)[None, :]
win = (wy * wx).astype(np.float32)

PG = np.zeros(NB); PK = np.zeros(NB); CC = np.zeros(NB)

# ---------- accumulators ----------
NMASK = 0
mask_names = []
acc_se = None; acc_n = None
gl_se = []; gl_var = []; gl_grad = []   # subsampled
SS = 11  # subsample stride
lum_bins = np.linspace(0, 1, 11)
lum_se = np.zeros(10); lum_n = np.zeros(10)
# global sorted-error concentration via fine log histogram
he = np.logspace(-8, 1.0, 700)
hist = np.zeros(len(he) + 1); hist_w = np.zeros(len(he) + 1)

tot_se = 0.0; tot_n = 0

# pass 1: thresholds on gradient / flatness from 8 GT images
gm_samp = []
for s in st[::8]:
    g = mlib.load_u8(os.path.join(gd, gm[s]))
    gg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    gx = cv2.Sobel(gg, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gg, cv2.CV_32F, 0, 1, 3)
    gm_samp.append(np.sqrt(gx * gx + gy * gy).ravel()[::5])
gm_samp = np.concatenate(gm_samp)
GQ = np.quantile(gm_samp, [0.5, 0.8, 0.95, 0.99])
print("GT sobel-grad quantiles p50/p80/p95/p99:", np.round(GQ, 4), flush=True)

MASKS = ["flat(<p50)", "low(p50-80)", "edge(p80-95)", "strong(p95-99)", "vstrong(>p99)",
         "sky", "nonsky", "near-strong-edge(<=3px)", "far-from-edge(>12px)"]
NM = len(MASKS)
m_se = np.zeros(NM); m_n = np.zeros(NM)
m_gtvar = np.zeros(NM); m_ensvar = np.zeros(NM)

for i, s in enumerate(st):
    G8 = mlib.load_u8(os.path.join(gd, gm[s]))
    K8 = apply_field(mlib.load_u8(os.path.join(K4, s + ".png")), FLD)
    M8 = [apply_field(mlib.load_u8(os.path.join(ROOT, m, "test_poses_renders_png", s + ".png")), FLD)
          for m in MEM]
    G = G8.astype(np.float32) / 255.; K = K8.astype(np.float32) / 255.
    Ms = np.stack([m.astype(np.float32) / 255. for m in M8])           # 4,H,W,3
    ensvar = Ms.var(0).mean(-1)                                        # H,W
    se = ((G - K) ** 2).mean(-1)                                       # H,W per-pixel MSE (RGB mean)
    tot_se += se.sum(); tot_n += se.size

    # --- spectrum on luminance ---
    gl = (cv2.cvtColor(G8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.) * win
    kl = (cv2.cvtColor(K8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.) * win
    Gf = np.fft.fft2(gl); Kf = np.fft.fft2(kl)
    pg = (Gf.real ** 2 + Gf.imag ** 2).ravel()
    pk = (Kf.real ** 2 + Kf.imag ** 2).ravel()
    cc = (Gf.real * Kf.real + Gf.imag * Kf.imag).ravel()
    PG += np.bincount(bidx, weights=pg, minlength=NB)
    PK += np.bincount(bidx, weights=pk, minlength=NB)
    CC += np.bincount(bidx, weights=cc, minlength=NB)

    # --- masks ---
    gg = cv2.cvtColor(G8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    gx = cv2.Sobel(gg, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gg, cv2.CV_32F, 0, 1, 3)
    gmag = np.sqrt(gx * gx + gy * gy)
    mu = cv2.blur(gg, (9, 9)); sd = np.sqrt(np.maximum(cv2.blur(gg * gg, (9, 9)) - mu * mu, 0))
    b, gch, r = G[..., 2], G[..., 1], G[..., 0]
    sky = (gg > 0.55) & (sd < 0.012) & (b >= r - 0.02)
    strong = (gmag > GQ[2]).astype(np.uint8)
    dist = cv2.distanceTransform(1 - strong, cv2.DIST_L2, 3)
    ml = [gmag < GQ[0], (gmag >= GQ[0]) & (gmag < GQ[1]), (gmag >= GQ[1]) & (gmag < GQ[2]),
          (gmag >= GQ[2]) & (gmag < GQ[3]), gmag >= GQ[3], sky, ~sky, dist <= 3, dist > 12]
    for j, mk in enumerate(ml):
        m_se[j] += se[mk].sum(); m_n[j] += mk.sum()
        m_gtvar[j] += (sd[mk] ** 2).sum(); m_ensvar[j] += ensvar[mk].sum()

    # --- luminance buckets ---
    li = np.clip((gg * 10).astype(int), 0, 9)
    lum_se += np.bincount(li.ravel(), weights=se.ravel(), minlength=10)
    lum_n += np.bincount(li.ravel(), minlength=10)

    # --- concentration histogram ---
    idx = np.digitize(se.ravel(), he)
    hist += np.bincount(idx, minlength=len(he) + 1)
    hist_w += np.bincount(idx, weights=se.ravel(), minlength=len(he) + 1)

    # --- subsample for correlation ---
    gl_se.append(se.ravel()[::SS].astype(np.float32))
    gl_var.append(ensvar.ravel()[::SS].astype(np.float32))
    gl_grad.append(gmag.ravel()[::SS].astype(np.float32))
    if i % 10 == 0: print("img", i, flush=True)

np.savez("/home/bkai/.claude/jobs/1c9cf7e9/tmp/B_out.npz",
         PG=PG, PK=PK, CC=CC, cnt=cnt, edges=edges, m_se=m_se, m_n=m_n,
         m_gtvar=m_gtvar, m_ensvar=m_ensvar, lum_se=lum_se, lum_n=lum_n,
         hist=hist, hist_w=hist_w, he=he, tot_se=tot_se, tot_n=tot_n,
         se=np.concatenate(gl_se), var=np.concatenate(gl_var), grad=np.concatenate(gl_grad),
         GQ=GQ, masks=np.array(MASKS))
print("done. total MSE", tot_se / tot_n, "-> PSNR", 10 * np.log10(1 / (tot_se / tot_n)))

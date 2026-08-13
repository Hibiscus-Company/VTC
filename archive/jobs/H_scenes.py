"""(d) What distinguishes our BEST from our WORST public scene? Residual stats, 5 scenes,
common variant = gsplatB9ut + train-fitted lens field (CPU only)."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(4)
SCENES = ["hcm0031", "HCM0193", "HCM0204", "HCM0181", "hcm0034"]   # worst -> best by score
H, W = 989, 1320
fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
rad = np.sqrt(fy ** 2 + fx ** 2); NB = 8
edges = np.linspace(0, 0.5, NB + 1)
bidx = np.clip(np.digitize(rad.ravel(), edges) - 1, 0, NB - 1)
cnt = np.bincount(bidx, minlength=NB).astype(float)
win = (np.hanning(H)[:, None] * np.hanning(W)[None, :]).astype(np.float32)
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
print(f"{'scene':9s} {'MSE':>9s} {'PSNR':>6s} {'GTgrad':>7s} {'GTHFpow':>9s} {'ampr_HF':>8s} "
      f"{'coh_HF':>7s} {'coh_LF':>7s} {'%SE<=3px':>9s} {'top1%SE':>8s} {'resid_flow':>10s} {'%SEsky':>7s}")
out = {}
for sc_ in SCENES:
    fld = np.load(f"/mnt/d/avv/fields/{sc_}.npy")
    rd = f"/mnt/d/avv/output/{sc_}_gsplatB9ut/test_poses_renders_png"
    gdir, gmp = gt_map(sc_)
    ss = [os.path.splitext(f)[0] for f in sorted(os.listdir(rd))]
    PG = np.zeros(NB); PK = np.zeros(NB); CC = np.zeros(NB)
    tse = 0.0; tn = 0; se_edge = 0.0; se_sky = 0.0; grads = 0.0; flows = []
    se_all = []
    for stem in ss:
        G8 = mlib.load_u8(os.path.join(gdir, gmp[stem]))
        K8 = apply_field(mlib.load_u8(os.path.join(rd, stem + ".png")), fld)
        G = G8.astype(np.float32) / 255.; K = K8.astype(np.float32) / 255.
        se = ((G - K) ** 2).mean(-1)
        tse += se.sum(); tn += se.size
        gg8 = cv2.cvtColor(G8, cv2.COLOR_RGB2GRAY); kk8 = cv2.cvtColor(K8, cv2.COLOR_RGB2GRAY)
        gray = gg8.astype(np.float32) / 255.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
        gmag = np.sqrt(gx * gx + gy * gy); grads += gmag.mean()
        thr = np.quantile(gmag, 0.95)
        dist = cv2.distanceTransform(1 - (gmag > thr).astype(np.uint8), cv2.DIST_L2, 3)
        se_edge += se[dist <= 3].sum()
        mu = cv2.blur(gray, (9, 9)); sd = np.sqrt(np.maximum(cv2.blur(gray * gray, (9, 9)) - mu * mu, 0))
        sky = (gray > 0.55) & (sd < 0.012) & (G[..., 2] >= G[..., 0] - 0.02)
        se_sky += se[sky].sum()
        gl = gray * win; kl = (kk8.astype(np.float32) / 255.) * win
        Gf = np.fft.fft2(gl); Kf = np.fft.fft2(kl)
        PG += np.bincount(bidx, weights=(Gf.real**2 + Gf.imag**2).ravel(), minlength=NB)
        PK += np.bincount(bidx, weights=(Kf.real**2 + Kf.imag**2).ravel(), minlength=NB)
        CC += np.bincount(bidx, weights=(Gf.real*Kf.real + Gf.imag*Kf.imag).ravel(), minlength=NB)
        flows.append(np.sqrt((dis.calc(gg8, kk8, None) ** 2).sum(-1)).mean())
        se_all.append(se.ravel()[::7])
    se_all = np.concatenate(se_all); se_all.sort()
    k = int(0.99 * len(se_all))
    top1 = se_all[k:].sum() / se_all.sum()
    hf = slice(4, 8); lf = slice(0, 2)
    def agg(sl):
        c = cnt[sl]
        pg = (PG[sl]).sum(); pk = (PK[sl]).sum(); cc = (CC[sl]).sum()
        return np.sqrt(pk / pg), cc / np.sqrt(pg * pk)
    ar_hf, coh_hf = agg(hf); _, coh_lf = agg(lf)
    hfpow = PG[hf].sum() / cnt[hf].sum()
    mse = tse / tn
    print(f"{sc_:9s} {mse:9.6f} {10*np.log10(1/mse):6.3f} {grads/len(ss):7.4f} {hfpow:9.3e} "
          f"{ar_hf:8.4f} {coh_hf:7.4f} {coh_lf:7.4f} {100*se_edge/tse:9.2f} {100*top1:8.2f} "
          f"{np.mean(flows):10.3f} {100*se_sky/tse:7.2f}", flush=True)

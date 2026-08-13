"""R4: spatial attribution of LPIPS itself + region spectra of the residual."""
import sys, json, numpy as np, torch, torch.nn.functional as F, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *
import lpips as lpips_pkg

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
H, W, N = 989, 1320, 60
se_all = np.load(OUT + "/se_all.npy")
var_all = np.load(OUT + "/var_all.npy")
code_all = np.load(OUT + "/code_all.npy")
m = lp().m


@torch.no_grad()
def lpips_map(a, b):
    """per-pixel LPIPS density; its spatial mean == the LPIPS scalar."""
    i0, i1 = m.scaling_layer(a * 2 - 1), m.scaling_layer(b * 2 - 1)
    f0, f1 = m.net.forward(i0), m.net.forward(i1)
    acc = torch.zeros(1, 1, a.shape[2], a.shape[3], device=a.device)
    tot = 0.0
    for k in range(m.L):
        d = (lpips_pkg.normalize_tensor(f0[k]) - lpips_pkg.normalize_tensor(f1[k])) ** 2
        v = m.lins[k](d)              # [1,1,hk,wk]
        tot += float(v.mean())
        acc += F.interpolate(v, size=(a.shape[2], a.shape[3]), mode='nearest') * \
            (1.0)  # nearest keeps mean ~ layer mean
    return acc[0, 0].cpu().numpy(), tot


lp_all = np.zeros((N, H, W), np.float32)
tots = []
for i in range(N):
    g = G[i].astype(np.float32) / 255.0
    p = apply_field(R[k4i, i].astype(np.float32) / 255.0, field)
    mp, tot = lpips_map(t(p), t(g))
    lp_all[i] = mp; tots.append(tot)
np.save(OUT + "/lp_all.npy", lp_all)
print("mean LPIPS from maps %.5f (scalar ref 0.10336); map mean %.5f"
      % (np.mean(tots), lp_all.mean()))

print("\n== concentration of LPIPS density vs SE ==")
print(" top-k%   LPIPS-share   SE-share")
for k in [0.5, 1, 2, 5, 10, 25]:
    cl, cs = 0.0, 0.0
    for i in range(N):
        n = int(k / 100 * H * W)
        l = np.sort(lp_all[i].ravel())[::-1]; s = np.sort(se_all[i].ravel())[::-1]
        cl += l[:n].sum() / l.sum(); cs += s[:n].sum() / s.sum()
    print(f" {k:5.1f}    {100*cl/N:8.2f}   {100*cs/N:8.2f}")

print("\n== LPIPS density / SE by region (relative to image mean) ==")
print(" region        px%    SE%   LPIPS%   relSE   relLPIPS  LPIPSper-SE")
tp = N * H * W
for cid, nm in [(1, "sky"), (2, "flat_nonsky"), (3, "midtex"), (4, "hightex")]:
    mk = code_all == cid
    px = mk.sum()
    ss = se_all[mk].sum(); ll = lp_all[mk].sum()
    print(f" {nm:12s} {100*px/tp:5.2f} {100*ss/se_all.sum():6.2f} {100*ll/lp_all.sum():7.2f}"
          f"   {(ss/px)/(se_all.sum()/tp):6.2f}x {(ll/px)/(lp_all.sum()/tp):8.2f}x"
          f"  {(ll/ss)/(lp_all.sum()/se_all.sum()):8.2f}x")

# correlations
sub = np.random.RandomState(0).choice(N * H * W, 400000, replace=False)
lf = lp_all.ravel()[sub]; sf = se_all.ravel()[sub]; vf = var_all.ravel()[sub]
from scipy.stats import spearmanr
print("\nSpearman(lpips_density, se)  %.4f" % spearmanr(lf, sf).correlation)
print("Spearman(lpips_density, var) %.4f" % spearmanr(lf, vf).correlation)
print("Spearman(se, var)            %.4f" % spearmanr(sf, vf).correlation)

# --- region spectra of the residual (is flat-region error HF or LF?) ---
print("\n== residual spectrum inside each region (patch DCT bands, share of resid power) ==")
bands = [(0, 2), (2, 4), (4, 8), (8, 16)]  # cycles per 16px patch
from scipy.fft import dctn
acc = {nm: np.zeros(len(bands)) for nm in ["sky", "flat_nonsky", "midtex", "hightex"]}
gtacc = {nm: np.zeros(len(bands)) for nm in acc}
for i in range(0, N, 4):
    g = G[i].astype(np.float32) / 255.0
    p = apply_field(R[k4i, i].astype(np.float32) / 255.0, field)
    y = (g * np.array([.299, .587, .114], np.float32)).sum(2)
    yp = (p * np.array([.299, .587, .114], np.float32)).sum(2)
    r = yp - y
    hh, ww = H // 16 * 16, W // 16 * 16
    def blocks(x):
        return x[:hh, :ww].reshape(hh // 16, 16, ww // 16, 16).transpose(0, 2, 1, 3)
    Br = dctn(blocks(r), axes=(2, 3), norm='ortho')
    Bg = dctn(blocks(y - y.mean()), axes=(2, 3), norm='ortho')
    cd = blocks(code_all[i].astype(np.float32)).mean((2, 3))
    u, v = np.mgrid[0:16, 0:16]
    rr = np.sqrt(u ** 2 + v ** 2)
    for cid, nm in [(1, "sky"), (2, "flat_nonsky"), (3, "midtex"), (4, "hightex")]:
        sel = np.abs(cd - cid) < 0.05
        if sel.sum() < 10:
            continue
        pr_ = (Br[sel] ** 2).mean(0); pg_ = (Bg[sel] ** 2).mean(0)
        for bi, (lo, hi) in enumerate(bands):
            mm = (rr >= lo) & (rr < hi)
            acc[nm][bi] += pr_[mm].sum(); gtacc[nm][bi] += pg_[mm].sum()
print(" region        resid: <2  2-4  4-8  8-16   |  gt: <2  2-4  4-8  8-16  (% of own power)")
for nm in acc:
    a = acc[nm] / acc[nm].sum() * 100; b = gtacc[nm] / gtacc[nm].sum() * 100
    print(f" {nm:12s} {a[0]:6.1f}{a[1]:6.1f}{a[2]:6.1f}{a[3]:6.1f}   | "
          f"{b[0]:6.1f}{b[1]:6.1f}{b[2]:6.1f}{b[3]:6.1f}")

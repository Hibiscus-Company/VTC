"""Exact analytic MSE/PSNR prediction for 'k4 base + member x at weight w', from the Gram data.
Gives the MSE-optimal w and predicted dPSNR for every candidate addition -- free, no GPU --
so we can see how much of the measured SCORE delta is PSNR vs LPIPS/SSIM."""
import os, json, numpy as np

OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens"
Z = np.load(os.path.join(OUT, "gram.npz"), allow_pickle=True)
A_all, b_all, c_all, n_all = Z["A"], Z["b"], Z["c"], Z["n"]
NAMES = list(Z["names"]); IDX = {n: i for i, n in enumerate(NAMES)}
singles = json.load(open(os.path.join(OUT, "singles.json")))
corr = {r["name"]: r for r in json.load(open(os.path.join(OUT, "corr.json")))}
V = len(NAMES)
K4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
MODELS = [m for m in NAMES if m not in ("sh3", "k4")]

# per-image residual inner products
Rip = A_all - b_all[:, :, None] - b_all[:, None, :] + c_all[:, None, None]   # (60,V,V)
w4 = np.zeros(V)
for m in K4:
    w4[IDX[m]] = 0.25


def psnr_mix(x, w):
    """per-image-mean PSNR of (1-w)*k4 + w*x"""
    i = IDX[x]
    ee = np.einsum("a,iab,b->i", w4, Rip, w4)
    ex = np.einsum("a,ia->i", w4, Rip[:, :, i])
    xx = Rip[:, i, i]
    mse = ((1 - w) ** 2 * ee + 2 * w * (1 - w) * ex + w ** 2 * xx) / n_all
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-12))))


base = psnr_mix(K4[0], 0.0)
print(f"analytic base k4 PSNR = {base:.4f}\n")
print(f"{'add':20s} {'single':>8s} {'corr':>6s} {'rmsRatio':>8s} {'w*(MSE)':>8s} {'dPSNR@w*':>9s} "
      f"{'dPSNR@1/5':>10s} {'dScorePSNRonly':>15s}")
rows = []
for x in MODELS:
    if x in K4:
        continue
    ws = np.linspace(0, 0.6, 121)
    ps = [psnr_mix(x, w) for w in ws]
    k = int(np.argmax(ps))
    d5 = psnr_mix(x, 0.2) - base
    rows.append(dict(add=x, wstar=float(ws[k]), dpsnr_star=ps[k] - base, dpsnr_02=d5))
    print(f"{x:20s} {singles[x]['score']:8.4f} {corr[x]['corr_k4']:6.3f} "
          f"{corr[x]['res_rms']/0.0609:8.2f} {ws[k]:8.3f} {ps[k]-base:+9.4f} {d5:+10.4f} "
          f"{0.6*d5:+15.4f}")
json.dump(rows, open(os.path.join(OUT, "theory.json"), "w"), indent=1)

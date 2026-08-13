"""P5: LOCAL CONTRAST MATCHING -- the SSIM identity nobody used.

Write the repo SSIM as l * c * s with the SAME 11x11 gaussian(1.5) window it uses.
Multiply the render's zero-mean local component by a gain g:
    sigma1 -> g*sigma1,  sigma12 -> g*sigma12
  => s = (sigma12 + C2/2)/(sigma1 sigma2 + C2/2)  is INVARIANT in g  (exactly, mod C2)
  => c = (2 sigma1 sigma2 + C2)/(sigma1^2 + sigma2^2 + C2)  is MAXIMISED at g = sigma2/sigma1
  => l is untouched.
So a local gain buys the whole c-deficit for free in SSIM terms; the only cost is PSNR
(the MSE-optimal gain is rho * sigma2/sigma1, a factor rho smaller) and whatever LPIPS does.

The gain is predicted from sigma1 ALONE via a lookup fitted on fold A and applied to fold B --
at production time the same lookup is fitted on TRAIN renders vs TRAIN photos. Rule 10 clean.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import mlib

torch.set_num_threads(int(os.environ.get("NT", "6")))
WS, C2 = 11, 0.03 ** 2
EDGES = np.array([0, .0025, .005, .01, .02, .04, .08, .16, 1.01])


def local(x, ws=WS):
    ch = x.shape[1]
    w = mlib.window(ws, ch)
    p = ws // 2
    mu = F.conv2d(x, w, padding=p, groups=ch)
    s2 = (F.conv2d(x * x, w, padding=p, groups=ch) - mu ** 2).clamp(min=1e-12)
    return mu, s2


def fit_lookup(pairs):
    """per-sigma1-bin  E[sigma2]/E[sigma1]  and the MSE-optimal gain, from (render,gt) tensors"""
    n = len(EDGES) - 1
    A = np.zeros(n); B = np.zeros(n); D = np.zeros(n); E = np.zeros(n)
    for r, g in pairs:
        _, s1 = local(r); _, s2 = local(g)
        mu1, _ = local(r); mu2, _ = local(g)
        w = mlib.window(WS, 3)
        s12 = F.conv2d(r * g, w, padding=WS // 2, groups=3) - mu1 * mu2
        sd1 = s1.sqrt().numpy().ravel(); sd2 = s2.sqrt().numpy().ravel()
        idx = np.clip(np.digitize(sd1, EDGES) - 1, 0, n - 1)
        np.add.at(A, idx, sd1); np.add.at(B, idx, sd2)
        np.add.at(D, idx, s12.numpy().ravel()); np.add.at(E, idx, s1.numpy().ravel())
    r_ssim = np.where(A > 0, B / np.maximum(A, 1e-12), 1.0)
    g_mse = np.where(E > 0, D / np.maximum(E, 1e-12), 1.0)
    return np.clip(r_ssim, 1.0, 4.0), np.clip(g_mse, 0.2, 4.0)


def regain(x, lut, lam, gmax=3.0):
    mu, s2 = local(x)
    sd = s2.sqrt()
    idx = np.clip(np.digitize(sd.numpy(), EDGES) - 1, 0, len(EDGES) - 2)
    g = torch.from_numpy(lut[idx]).float()
    g = 1.0 + lam * (g - 1.0)
    g = g.clamp(1.0 / gmax, gmax)
    # smooth the gain map so it does not create its own structure
    w = mlib.window(WS, 3)
    g = F.conv2d(g, w, padding=WS // 2, groups=3)
    return (mu + g * (x - mu)).clamp(0, 1)


key = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CASES = {
    "HCM0421": ("/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/evalsplit/HCM0421/eval_gt",
                ["DJI_20241230093301_0003_V"]),
    "chair": ("/mnt/d/avv/chair_eval/base60k/eval_png", "/mnt/d/avv/evalsplit/chair/eval_gt", []),
    "bonsai": ("/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_png", "/mnt/d/avv/evalsplit/bonsai2/eval_gt", []),
}
rd, gd, skip = CASES[key]
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
files = [f for f in sorted(os.listdir(rd)) if os.path.splitext(f)[0] not in skip]
files = files[:: max(1, len(files) // N)][:N]
R = [mlib.to_t(mlib.load_u8(os.path.join(rd, f))) for f in files]
G = [mlib.to_t(mlib.load_u8(os.path.join(gd, gt_by[os.path.splitext(f)[0]]))) for f in files]

lp = mlib.LP("cpu")
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
names = [f"ssimlut_lam{l}" for l in LAMS] + ["mselut_lam1.0", "flat_g1.10", "flat_g1.25"]
acc = {n: [0.0, 0.0, 0.0, 0] for n in names}
A = list(range(0, len(files), 2)); B = list(range(1, len(files), 2))
for fitset, testset in ((A, B), (B, A)):
    lut_s, lut_m = fit_lookup([(R[i], G[i]) for i in fitset])
    print("  sigma-bin gain lookup (SSIM-opt):", np.round(lut_s, 3),
          "\n  (MSE-opt):", np.round(lut_m, 3), flush=True)
    for i in testset:
        g = G[i]
        outs = {f"ssimlut_lam{l}": regain(R[i], lut_s, l) for l in LAMS}
        outs["mselut_lam1.0"] = regain(R[i], lut_m, 1.0)
        for gg in (1.10, 1.25):
            outs[f"flat_g{gg:.2f}"] = regain(R[i], np.full(len(EDGES) - 1, gg), 1.0)
        for n in names:
            y = outs[n]
            acc[n][0] += mlib.psnr(y, g); acc[n][1] += float(mlib.ssim(y, g))
            acc[n][2] += lp(y, g); acc[n][3] += 1
        print(f"   img {i} done", flush=True)

print(f"\n=== P5 {key} n={len(files)} 2-fold CV local-contrast gain ===")
rows = [(n, acc[n][0] / acc[n][3], acc[n][1] / acc[n][3], acc[n][2] / acc[n][3]) for n in names]
rows = [(n, P, S, L, mlib.score(P, S, L)) for n, P, S, L in rows]
ref = rows[0]
for n, P, S, L, sc in rows:
    print(f"{n:16s} PSNR {P:7.4f}({P-ref[1]:+.4f})  SSIM {S:.5f}({S-ref[2]:+.5f})  "
          f"LPIPS {L:.5f}({L-ref[3]:+.5f})  SCORE {sc:8.4f}  d {sc-ref[4]:+.4f}")
json.dump(rows, open(f"/mnt/d/avv/metric_probe/p5_{key}.json", "w"), indent=1)

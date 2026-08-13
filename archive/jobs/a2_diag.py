#!/usr/bin/env python
"""Extra diagnostics: does the RENDER already track the local blur average?
Bootstrap CIs on the small-n (28) R2 values.  Score decomposition."""
import sys, csv, json
import numpy as np
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from blur_predictor import BlurPredictor, STRIDE
from scipy.stats import spearmanr

OUT = "/mnt/d/avv/r42_bonsai78/a2_blurpred"
EVAL = sorted(int(x) for x in json.load(open("/mnt/d/avv/evalsplit/bonsai/split.json"))["eval_frames"])

def load_csv(p):
    return {int(r["frame"]): {k: float(v) for k, v in r.items() if k != "name"}
            for r in csv.DictReader(open(p))}

T = load_csv(f"{OUT}/train248_sharp.csv"); R = load_csv(f"{OUT}/render_sr01_sharp.csv")
S = load_csv(f"{OUT}/perframe_sr01.csv")
SUB = [f for f in sorted(T) if f not in set(EVAL)]

def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

P = {}
for st, mode in [("log_lapvar", "nw"), ("log_hf025", "nw"), ("reblur", "ll")]:
    bp = BlurPredictor.fit({f: T[f][st] for f in SUB}, mode=mode)
    P[st] = dict(h=bp.h, both=bp.predict_many(EVAL),
                 one=bp.predict_many(EVAL, {f: (f + STRIDE,) for f in EVAL}))
    print(f"chosen {st:11s} mode={mode} h={bp.h:.0f} loo_r2(train_sub)={bp.loo_r2:.3f}")

gt = np.array([T[f]["log_lapvar"] for f in EVAL])
rn = np.array([R[f]["log_lapvar"] for f in EVAL])
pr = P["log_lapvar"]["both"]
sc = np.array([S[f]["score"] for f in EVAL])
lp = np.array([S[f]["lpips"] for f in EVAL])
ps = np.array([S[f]["psnr"] for f in EVAL])
ss = np.array([S[f]["ssim"] for f in EVAL])

print("\n--- is the render already tracking the LOCAL BLUR AVERAGE? ---")
print(f"  corr(GT, render)        = {np.corrcoef(gt, rn)[0,1]:+.3f}")
print(f"  corr(PRED_gt, render)   = {np.corrcoef(pr, rn)[0,1]:+.3f}   "
      f"(PRED_gt uses only the neighbours -> pure local blur average)")
# partial corr(GT, render | PRED)
def resid(y, x):
    A = np.column_stack([np.ones(len(x)), x]); c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ c
print(f"  partial corr(GT, render | PRED_gt) = "
      f"{np.corrcoef(resid(gt, pr), resid(rn, pr))[0,1]:+.3f}  "
      f"(this part is CONTENT, not blur)")
print(f"  R2 of render_sharp explained by PRED_gt alone: {r2(rn, np.polyval(np.polyfit(pr, rn, 1), pr)):.3f}")

print("\n--- bootstrap 95% CI over the 28 holes (2000 resamples) ---")
rng = np.random.default_rng(0)
def boot(y, p, n=2000):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yy, pp = y[i], p[i]
        if yy.std() < 1e-9: continue
        v.append(1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2))
    return np.percentile(v, [2.5, 97.5])
for nm, y, p in [("sharpness log_lapvar (both nb)", gt, P["log_lapvar"]["both"]),
                 ("sharpness log_lapvar (one nb)", gt, P["log_lapvar"]["one"]),
                 ("sharpness reblur (both nb)", np.array([T[f]["reblur"] for f in EVAL]), P["reblur"]["both"]),
                 ("log-ratio GT/render", gt - rn, P["log_lapvar"]["both"] - rn)]:
    lo, hi = boot(y, p)
    print(f"  {nm:32s} R2 = {r2(y,p):+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")

print("\n--- STEP 4 correlations with the per-frame score ---")
tab = [("ACTUAL log_lapvar (oracle)", gt), ("PRED log_lapvar (legal, both nb)", pr),
       ("PRED log_lapvar (legal, one nb)", P["log_lapvar"]["one"]),
       ("ACTUAL reblur", np.array([T[f]["reblur"] for f in EVAL])),
       ("PRED reblur", P["reblur"]["both"]),
       ("RENDER log_lapvar", rn),
       ("ACTUAL log-ratio", gt - rn), ("PRED log-ratio", pr - rn)]
print(f"  {'variable':34s} {'r(score)':>9s} {'rho(score)':>10s} {'r(lpips)':>9s} "
      f"{'r(psnr)':>8s} {'r(ssim)':>8s}")
for nm, x in tab:
    print(f"  {nm:34s} {np.corrcoef(x,sc)[0,1]:+9.3f} {spearmanr(x,sc).statistic:+10.3f} "
          f"{np.corrcoef(x,lp)[0,1]:+9.3f} {np.corrcoef(x,ps)[0,1]:+8.3f} {np.corrcoef(x,ss)[0,1]:+8.3f}")

print("\n--- how much per-frame score variance is 'explained' by sharpness? ---")
for nm, x in tab:
    b = np.polyfit(x, sc, 1); f = np.polyval(b, x)
    print(f"  {nm:34s} slope {b[0]:+8.3f} pts/unit  R2 {r2(sc,f):+.3f}  "
          f"resid sd {(sc-f).std(ddof=1):.3f} (tot sd {sc.std(ddof=1):.3f})")

# Fisher CI on a correlation with n=28
def rci(r, n=28):
    z = np.arctanh(r); s = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * s), np.tanh(z + 1.96 * s)
for nm, x in [("PRED log_lapvar", pr), ("ACTUAL log_lapvar", gt), ("PRED log-ratio", pr - rn)]:
    r = np.corrcoef(x, sc)[0, 1]; lo, hi = rci(r)
    print(f"  r({nm}, score) = {r:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]"
          f"{'   <-- crosses 0' if lo*hi<0 else ''}")

print("\n--- terciles of PREDICTED sharpness vs mean score (is it actionable?) ---")
o = np.argsort(pr)
for k, lab in [(o[:9], "blurriest-predicted 9"), (o[9:19], "middle 10"), (o[19:], "sharpest-predicted 9")]:
    print(f"  {lab:24s} mean score {sc[k].mean():7.3f}  mean PSNR {ps[k].mean():6.3f} "
          f"mean LPIPS {lp[k].mean():.4f}  mean actual log_lapvar {gt[k].mean():+.3f}")

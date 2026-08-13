"""How much of the per-frame ORACLE gain survives if the (sigma,alpha) must be chosen
without seeing that frame's GT?  Leave-one-out over the 28 eval frames:
  - LOO global: pick the setting that maximises the mean score of the OTHER 27 frames.
  - LOO regression on render-side lapvar: fit alpha ~ f(log lapvar_render) on the other 27.
"""
import os, json, glob
import numpy as np

CACHE = "/mnt/d/avv/r42_bonsai78/a1_oracle/cache"


def sc(p, s, l):
    return 100.0 * (0.4 * (1 - l) + 0.3 * s + 0.3 * min(p / 50.0, 1.0))


D = {}
for f in glob.glob(os.path.join(CACHE, "*.json")):
    r = json.load(open(f))
    D.setdefault(r["stem"], {})[r["key"]] = r
stems = sorted(D)
have = sorted(set.intersection(*[set(D[s]) for s in stems]))
keys = ["base"] + sorted([k for k in have if k.startswith("us_")],
                         key=lambda k: (float(k.split("_")[1]), float(k.split("_")[2])))
M = np.array([[sc(D[s][k]["psnr"], D[s][k]["ssim"], D[s][k]["lpips"]) for k in keys] for s in stems])
base = M[:, 0]
print("frames", len(stems), "settings", len(keys))
print("baseline mean score  %.4f" % base.mean())
gi = int(np.argmax(M.mean(0)))
print("global best  %-12s  %.4f  (d %+.4f)" % (keys[gi], M[:, gi].mean(), M[:, gi].mean() - base.mean()))
orc = M.max(1)
print("oracle       %.4f  (d %+.4f)" % (orc.mean(), orc.mean() - base.mean()))

# LOO global
loo = []
for i in range(len(stems)):
    m = np.delete(M, i, axis=0).mean(0)
    loo.append(M[i, int(np.argmax(m))])
loo = np.array(loo)
print("LOO global   %.4f  (d %+.4f)" % (loo.mean(), loo.mean() - base.mean()))

# LOO predictor: alpha from log lapvar_render (ordinal ridge on the argmax index)
lv = np.log(np.array([D[s]["base"]["lapvar_render"] for s in stems]))
lg = np.log(np.array([D[s]["base"]["lapvar_gt"] for s in stems]))
best_idx = M.argmax(1)
for name, x in (("lapvar_render", lv), ("lapvar_gt(cheat)", lg)):
    pred = []
    for i in range(len(stems)):
        tr = [j for j in range(len(stems)) if j != i]
        A = np.vstack([np.ones(len(tr)), x[tr]]).T
        coef, *_ = np.linalg.lstsq(A, best_idx[tr].astype(float), rcond=None)
        k = int(np.clip(round(coef[0] + coef[1] * x[i]), 0, len(keys) - 1))
        pred.append(M[i, k])
    pred = np.array(pred)
    print("LOO reg(%-16s) %.4f  (d %+.4f)" % (name, pred.mean(), pred.mean() - base.mean()))

print("\nper-setting mean score:")
for j, k in enumerate(keys):
    print("  %-12s %.4f  d %+.4f   frames-where-best %d" % (
        k, M[:, j].mean(), M[:, j].mean() - base.mean(), int((best_idx == j).sum())))

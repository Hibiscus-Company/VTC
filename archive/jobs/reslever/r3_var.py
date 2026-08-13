"""R3: is ensemble disagreement a usable per-pixel error predictor?
   plus oracle region-substitution -> how many SCORE points each error mode is worth."""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *
from scipy.stats import spearmanr

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
mids = [IDX[m] for m in MEM]
H, W, N = 989, 1320, 60
se_all = np.load(OUT + "/se_all.npy")
code_all = np.load(OUT + "/code_all.npy")

var_all = np.zeros((N, H, W), np.float32)
pr, sr, spr = [], [], []
for i in range(N):
    ms = np.stack([apply_field(R[m, i].astype(np.float32) / 255.0, field) for m in mids])
    v = ms.var(0).mean(2)
    var_all[i] = v
    se = se_all[i]
    a = np.log(v.ravel() + 1e-8); b = np.log(se.ravel() + 1e-8)
    pr.append(np.corrcoef(a, b)[0, 1])
    sub = np.random.RandomState(i).choice(v.size, 200000, replace=False)
    spr.append(spearmanr(v.ravel()[sub], se.ravel()[sub]).correlation)
    sr.append(np.corrcoef(v.ravel(), se.ravel())[0, 1])
np.save(OUT + "/var_all.npy", var_all)
print("corr(var,se)      raw Pearson  %.4f" % np.mean(sr))
print("corr(log var,log se) Pearson   %.4f" % np.mean(pr))
print("Spearman                        %.4f" % np.mean(spr))

# how much SE is captured by top-k% variance pixels vs the oracle top-k% SE pixels
print("\n top-k%% pixels selected by ...   SE captured")
print("  k%     by-variance   by-oracle-SE")
for k in [0.5, 1, 2, 5, 10, 25]:
    cv_, co = 0.0, 0.0
    for i in range(N):
        v = var_all[i].ravel(); se = se_all[i].ravel()
        n = int(k / 100 * v.size)
        idx = np.argpartition(v, -n)[-n:]
        cv_ += se[idx].sum() / se.sum()
        so = np.sort(se)[::-1]
        co += so[:n].sum() / se.sum()
    print(f" {k:5.1f}    {100*cv_/N:8.2f}      {100*co/N:8.2f}")

# variance decile -> mean SE
print("\n var decile -> relative mean SE")
rel = np.zeros(10)
for i in range(N):
    v = var_all[i]; se = se_all[i]
    thr = np.quantile(v, np.linspace(0, 1, 11)[1:-1])
    d = np.digitize(v, thr)
    for j in range(10):
        rel[j] += se[d == j].mean() / se.mean()
for j in range(10):
    print(f"  d{j}: {rel[j]/N:6.2f}x")

# ---------------- oracle substitution: score value of each error mode ----------------
base = None


def oracle(maskfn, tag, out):
    P = S = L = 0.0
    for i in range(N):
        g = G[i].astype(np.float32) / 255.0
        p = apply_field(R[k4i, i].astype(np.float32) / 255.0, field)
        m = maskfn(i)
        if m is not None:
            p = np.where(m[..., None], g, p)
        a, b, c = score(p, g)
        P += a; S += b; L += c
    P, S, L = P / N, S / N, L / N
    out[tag] = dict(psnr=P, ssim=S, lpips=L, comp=comp(P, S, L))
    print(f" {tag:22s} PSNR {P:7.4f} SSIM {S:.5f} LPIPS {L:.5f} comp {comp(P,S,L):8.4f}")
    return comp(P, S, L)


res = {}
print("\n== oracle substitution (replace mask with GT) ==")
b = oracle(lambda i: None, "baseline", res)


def topk_var(i, k):
    v = var_all[i].ravel()
    n = int(k / 100 * v.size)
    th = np.partition(v, -n)[-n]
    return var_all[i] >= th


def topk_se(i, k):
    v = se_all[i].ravel()
    n = int(k / 100 * v.size)
    th = np.partition(v, -n)[-n]
    return se_all[i] >= th


for k in [1, 5, 10, 25]:
    c = oracle(lambda i, k=k: topk_var(i, k), f"top{k}%_variance", res)
    print(f"      -> delta {c-b:+.4f}")
for k in [1, 5, 10, 25]:
    c = oracle(lambda i, k=k: topk_se(i, k), f"top{k}%_oracleSE", res)
    print(f"      -> delta {c-b:+.4f}")
for cid, nm in [(1, "sky"), (2, "flat_nonsky"), (3, "midtex"), (4, "hightex")]:
    c = oracle(lambda i, cid=cid: code_all[i] == cid, f"region_{nm}", res)
    print(f"      -> delta {c-b:+.4f}")
json.dump(res, open(OUT + "/r3.json", "w"))

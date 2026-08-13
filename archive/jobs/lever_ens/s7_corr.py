"""Residual structure from the Gram matrices: per-member residual RMS, and residual
correlation with (a) the best single model, (b) the k4 harness ensemble.
Everything exact, no sampling."""
import os, json, numpy as np

OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens"
Z = np.load(os.path.join(OUT, "gram.npz"), allow_pickle=True)
A_all, b_all, c_all, n_all = Z["A"], Z["b"], Z["c"], Z["n"]
NAMES = list(Z["names"]); IDX = {n: i for i, n in enumerate(NAMES)}
singles = json.load(open(os.path.join(OUT, "singles.json")))
A = A_all.sum(0); b = b_all.sum(0); c = c_all.sum(); n = n_all.sum()
V = len(NAMES)

# residual inner products:  <ri, rj> = <xi,xj> - <xi,y> - <xj,y> + <y,y>
Rip = A - b[:, None] - b[None, :] + c
rms = np.sqrt(np.diag(Rip) / n)
C = Rip / np.sqrt(np.outer(np.diag(Rip), np.diag(Rip)))

MODELS = [m for m in NAMES if m not in ("sh3", "k4")]
K4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
w4 = np.zeros(V);
for m in K4: w4[IDX[m]] = 0.25
# residual of ensemble = sum w_i r_i
r_ens_ip = w4 @ Rip @ w4
BEST = IDX["gsplatB11ut60k"]

print(f"{'member':20s} {'single':>8s} {'d_vs_best':>9s} {'resRMS':>7s} {'corr_best':>9s} {'corr_k4':>8s} {'unique%':>7s}")
rows = []
for m in MODELS:
    i = IDX[m]
    cb = C[i, BEST]
    ck = (w4 @ Rip[:, i]) / np.sqrt(r_ens_ip * Rip[i, i])
    uniq = 100 * (1 - ck ** 2)
    d = singles[m]["score"] - singles["gsplatB11ut60k"]["score"]
    print(f"{m:20s} {singles[m]['score']:8.4f} {d:+9.4f} {rms[i]:7.4f} {cb:9.3f} {ck:8.3f} {uniq:7.1f}")
    rows.append(dict(name=m, single=singles[m]["score"], d_vs_best=d, res_rms=float(rms[i]),
                     corr_best=float(cb), corr_k4=float(ck)))
json.dump(rows, open(os.path.join(OUT, "corr.json"), "w"), indent=1)

print("\n--- residual correlation matrix (models only) ---")
print(f"{'':20s}" + "".join(f"{m[:7]:>8s}" for m in MODELS))
for m in MODELS:
    print(f"{m:20s}" + "".join(f"{C[IDX[m],IDX[q]]:8.2f}" for q in MODELS))

import numpy as np
d = np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/B_out.npz", allow_pickle=True)
PG, PK, CC, cnt, edges = d['PG'], d['PK'], d['CC'], d['cnt'], d['edges']
ok = cnt > 0
pg, pk, cc, c = PG[ok] / cnt[ok], PK[ok] / cnt[ok], CC[ok] / cnt[ok], cnt[ok]
fc = (0.5 * (edges[:-1] + edges[1:]))[ok]
PR = pg + pk - 2 * cc
coh = cc / np.sqrt(pg * pk)
amp = (np.sqrt(pg) - np.sqrt(pk)) ** 2
inc = 2 * np.sqrt(pg * pk) * (1 - coh)
wien = pg - cc ** 2 / pk                       # MSE after optimal radial linear filter
tot = (PR * c).sum()
print("== (a) RADIAL POWER SPECTRUM: GT vs production render (k4+field), luminance, 60 imgs ==")
print("cyc/px   lam(px)  P_G          amp-ratio  coher   resid%%tot  amp-mism%%  incoh%%   linear-fixable%%")
grp = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 42), (42, 48)]
rows = []
for a, b in grp:
    sl = slice(a, b)
    cs = c[sl]
    f0, f1 = edges[a], edges[b]
    Pg = (pg[sl] * cs).sum() / cs.sum(); Pk = (pk[sl] * cs).sum() / cs.sum()
    Rr = (PR[sl] * cs).sum(); Am = (amp[sl] * cs).sum(); In = (inc[sl] * cs).sum()
    Wi = (wien[sl] * cs).sum()
    Co = (cc[sl] * cs).sum() / np.sqrt(((pg[sl] * cs).sum()) * ((pk[sl] * cs).sum()))
    print(f"{f0:.3f}-{f1:.3f} {1/max(f1,1e-9):5.1f}-{1/max(f0,1e-9) if f0>0 else 999:<5.0f} "
          f"{Pg:11.3e} {np.sqrt(Pk/Pg):8.4f}  {Co:6.4f}  {100*Rr/tot:8.2f}  "
          f"{100*Am/tot:8.2f}  {100*In/tot:7.2f}  {100*(Rr-Wi)/tot:8.2f}")
    rows.append((f0, f1, Pg, np.sqrt(Pk / Pg), Co, Rr / tot, Am / tot, In / tot, (Rr - Wi) / tot))
print(f"TOTAL amp-mismatch {100*(amp*c).sum()/tot:.2f}%  incoherent {100*(inc*c).sum()/tot:.2f}%"
      f"  max recoverable by ANY radial linear filter {100*(tot-(wien*c).sum())/tot:.2f}%")

print("\n== (b) ERROR CONCENTRATION ==")
hist, hw, he = d['hist'], d['hist_w'], d['he']
o = np.argsort(-np.arange(len(hist)))     # descending bin index = descending error
cn = np.cumsum(hist[o]); cw = np.cumsum(hw[o])
N = hist.sum(); T = hw.sum()
for frac in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    k = np.searchsorted(cn, frac * N)
    print(f"  top {100*frac:5.2f}% of pixels carry {100*cw[min(k,len(cw)-1)]/T:5.1f}% of total squared error")

print("\n== masks (per-pixel MSE, RGB-mean) ==")
names = list(d['masks']); m_se, m_n = d['m_se'], d['m_n']
base = m_se.sum() / m_n[6] if False else d['tot_se'] / d['tot_n']
print(f"  {'mask':26s} {'%px':>6s} {'MSE':>10s} {'PSNR':>7s} {'%totSE':>7s} {'ensSD(1e-3)':>11s}")
TOT = float(d['tot_se'])
for i, nm in enumerate(names):
    if m_n[i] == 0: continue
    mse = m_se[i] / m_n[i]
    print(f"  {nm:26s} {100*m_n[i]/d['tot_n']:6.2f} {mse:10.6f} {10*np.log10(1/mse):7.3f} "
          f"{100*m_se[i]/TOT:7.2f} {1e3*np.sqrt(d['m_ensvar'][i]/m_n[i]):11.3f}")
print(f"  {'ALL':26s} {100.0:6.2f} {base:10.6f} {10*np.log10(1/base):7.3f} {100.0:7.2f}")

print("\n== luminance buckets ==")
ls, ln = d['lum_se'], d['lum_n']
for i in range(10):
    if ln[i] == 0: continue
    print(f"  Y {i/10:.1f}-{(i+1)/10:.1f}  {100*ln[i]/d['tot_n']:6.2f}%px  MSE {ls[i]/ln[i]:.6f}  %totSE {100*ls[i]/TOT:6.2f}")

print("\n== (c) ENSEMBLE VARIANCE vs ERROR ==")
se, var, grad = d['se'].astype(np.float64), d['var'].astype(np.float64), d['grad'].astype(np.float64)
print(f"  n={len(se)}  mean se {se.mean():.6f}  mean var {var.mean():.6e}")
def r(a, b): return np.corrcoef(a, b)[0, 1]
print(f"  pearson(se, var)          = {r(se,var):.4f}")
print(f"  pearson(sqrt se, sqrt var)= {r(np.sqrt(se),np.sqrt(var)):.4f}")
print(f"  pearson(log se, log var)  = {r(np.log(se+1e-9),np.log(var+1e-12)):.4f}")
print(f"  pearson(se, grad^2)       = {r(se,grad**2):.4f}")
print(f"  pearson(sqrt se, grad)    = {r(np.sqrt(se),grad):.4f}")
# how well does var predict se (R^2 of best 1-D monotone/binned predictor)
q = np.quantile(var, np.linspace(0, 1, 21))
bi = np.clip(np.digitize(var, q[1:-1]), 0, 19)
mus = np.array([se[bi == k].mean() for k in range(20)])
pred = mus[bi]
R2 = 1 - ((se - pred) ** 2).sum() / ((se - se.mean()) ** 2).sum()
print(f"  R^2 of 20-bin var->E[se] predictor: {R2:.4f}")
qg = np.quantile(grad, np.linspace(0, 1, 21)); bg = np.clip(np.digitize(grad, qg[1:-1]), 0, 19)
mug = np.array([se[bg == k].mean() for k in range(20)]); R2g = 1 - ((se - mug[bg]) ** 2).sum() / ((se - se.mean()) ** 2).sum()
print(f"  R^2 of 20-bin grad->E[se] predictor: {R2g:.4f}")
# 2D
bb = bi * 20 + bg
mu2 = np.zeros(400)
for k in range(400):
    m = bb == k
    mu2[k] = se[m].mean() if m.sum() > 30 else se.mean()
R22 = 1 - ((se - mu2[bb]) ** 2).sum() / ((se - se.mean()) ** 2).sum()
print(f"  R^2 of 20x20 (var,grad)->E[se] predictor: {R22:.4f}")
print("\n  var-decile   meanVar      meanSE       SE/overall   share%totSE")
for k in range(0, 20, 2):
    m = (bi == k) | (bi == k + 1)
    print(f"   {k//2+1:2d}         {var[m].mean():.3e}  {se[m].mean():.6f}   "
          f"{se[m].mean()/se.mean():7.2f}x   {100*se[m].sum()/se.sum():6.2f}")

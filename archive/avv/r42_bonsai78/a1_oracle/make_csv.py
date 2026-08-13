import os, json, glob, csv
import numpy as np
CACHE = "/mnt/d/avv/r42_bonsai78/a1_oracle/cache"
OUT = "/mnt/d/avv/r42_bonsai78/a1_oracle/per_frame.csv"
SPEC_SIG = [0.6, 1.0, 1.6, 2.5]
SPEC_A = [0.15, 0.3, 0.5, 0.8, 1.2]


def sc(p, s, l):
    return 100.0 * (0.4 * (1 - l) + 0.3 * s + 0.3 * min(p / 50.0, 1.0))


D = {}
for f in glob.glob(os.path.join(CACHE, "*.json")):
    r = json.load(open(f))
    D.setdefault(r["stem"], {})[r["key"]] = r
stems = sorted(D)
spec_keys = ["base"] + [f"us_{s}_{a}" for s in SPEC_SIG for a in SPEC_A]
ext_keys = sorted([k for k in D[stems[0]] if k.startswith("us_")],
                  key=lambda k: (float(k.split("_")[1]), float(k.split("_")[2]))) + ["base"]


def frow(s, k):
    d = D[s][k]
    return sc(d["psnr"], d["ssim"], d["lpips"]), d["psnr"], d["ssim"], d["lpips"]


rows = []
for s in stems:
    bs, bp, bss, bl = frow(s, "base")
    lg, lr = D[s]["base"]["lapvar_gt"], D[s]["base"]["lapvar_render"]
    k1 = max(spec_keys, key=lambda k: frow(s, k)[0])
    k2 = max(ext_keys, key=lambda k: frow(s, k)[0])
    o1 = frow(s, k1)
    o2 = frow(s, k2)
    sp = frow(s, "spec_pc")
    ss_ = frow(s, "spec_sh")
    sw = frow(s, "specw_pc")
    sg1, a1 = ("0", "0") if k1 == "base" else k1.split("_")[1:]
    sg2, a2 = ("0", "0") if k2 == "base" else k2.split("_")[1:]
    rows.append(dict(
        stem=s, base_score=bs, base_psnr=bp, base_ssim=bss, base_lpips=bl,
        lapvar_gt=lg, lapvar_render=lr, lapvar_ratio_gt_over_render=lg / lr,
        oracle_sigma=sg1, oracle_alpha=a1, oracle_score=o1[0], oracle_dscore=o1[0] - bs,
        oracle_dpsnr=o1[1] - bp, oracle_dssim=o1[2] - bss, oracle_dlpips=o1[3] - bl,
        oracle_ext_sigma=sg2, oracle_ext_alpha=a2, oracle_ext_score=o2[0],
        oracle_ext_dscore=o2[0] - bs, oracle_ext_dpsnr=o2[1] - bp,
        oracle_ext_dssim=o2[2] - bss, oracle_ext_dlpips=o2[3] - bl,
        specmatch_pc_score=sp[0], specmatch_pc_dscore=sp[0] - bs, specmatch_pc_dpsnr=sp[1] - bp,
        specmatch_pc_dssim=sp[2] - bss, specmatch_pc_dlpips=sp[3] - bl,
        specmatch_sh_score=ss_[0], specmatch_sh_dscore=ss_[0] - bs,
        specwiener_pc_score=sw[0], specwiener_pc_dscore=sw[0] - bs,
        specwiener_pc_dpsnr=sw[1] - bp, specwiener_pc_dssim=sw[2] - bss,
        specwiener_pc_dlpips=sw[3] - bl,
        bestglobal_specgrid_key="base(a=0)", bestglobal_specgrid_dscore=0.0,
        bestglobal_ext_key="us_1.6_-0.15",
        bestglobal_ext_dscore=frow(s, "us_1.6_-0.15")[0] - bs))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
print("wrote", OUT, len(rows), "rows")

# summary of per-frame spread
d1 = np.array([r["oracle_dscore"] for r in rows])
d2 = np.array([r["oracle_ext_dscore"] for r in rows])
d3 = np.array([r["specmatch_pc_dscore"] for r in rows])
d4 = np.array([r["specwiener_pc_dscore"] for r in rows])
db = np.array([r["bestglobal_ext_dscore"] for r in rows])
for n, d in (("oracle(spec grid)", d1), ("oracle(ext,incl blur)", d2),
             ("specmatch_pc", d3), ("specwiener_pc", d4), ("global us_1.6_-0.15", db)):
    print("%-24s mean %+.4f  min %+.4f  max %+.4f  frames>0 %d/28" % (
        n, d.mean(), d.min(), d.max(), int((d > 0).sum())))
lg = np.array([r["lapvar_gt"] for r in rows])
lr = np.array([r["lapvar_render"] for r in rows])
bs = np.array([r["base_score"] for r in rows])
from scipy.stats import spearmanr
print("\nlapvar_gt: min %.3e p50 %.3e max %.3e  (spread %.1fx)" % (lg.min(), np.median(lg), lg.max(), lg.max() / lg.min()))
print("lapvar_ren: min %.3e p50 %.3e max %.3e (spread %.1fx)" % (lr.min(), np.median(lr), lr.max(), lr.max() / lr.min()))
print("ratio gt/ren: min %.2f p50 %.2f max %.2f" % ((lg / lr).min(), np.median(lg / lr), (lg / lr).max()))
print("spearman(base_score, lapvar_gt) = %.3f" % spearmanr(bs, lg).correlation)
print("spearman(base_score, lapvar_ratio) = %.3f" % spearmanr(bs, lg / lr).correlation)
print("spearman(oracle_ext_dscore, lapvar_gt) = %.3f" % spearmanr(d2, lg).correlation)
ea = np.array([float(r["oracle_ext_alpha"]) for r in rows])
print("spearman(oracle_ext_alpha, lapvar_gt) = %.3f  (positive => sharper GT wants larger alpha)" % spearmanr(ea, lg).correlation)
print("spearman(oracle_ext_alpha, lapvar_ratio gt/ren) = %.3f" % spearmanr(ea, lg / lr).correlation)

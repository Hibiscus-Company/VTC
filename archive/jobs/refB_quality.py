#!/usr/bin/env python
"""REFUTATION STEP B: is s2gates champA a GOOD member, or just a DIFFERENT one?

Deviation from the pool mean cannot tell those apart. Two GT-free instruments that can:

 (1) FAMILY-INTERNAL SPREAD. champA/memB/memC are three trainings of one recipe; ut7/13/42/77/
     seed101 are five of another. The pairwise spread WITHIN a family measures that recipe's own
     run-to-run noise -- and run-to-run noise is a direct lower bound on per-member error. If the
     s2gates family's internal spread is far larger than UT's, the recipe is noisier, i.e. worse,
     and the extra "decorrelation" the claim sells is that noise, not independent signal.

 (2) CONSENSUS-DEVIATION QUALITY PROXY, VALIDATED ON REAL GT. On HCM0181 solo PSNR is known for
     all 21 variants, so we can check whether deviation from a broad leave-one-out consensus
     predicts true member quality. If it does, we can run the same statistic on the private
     towers and read off where champA sits.
"""
import os, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
NV = int(os.environ.get("NV", "10"))
ld = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def load_all(dirs, stems, ext=".png"):
    return {n: np.stack([ld(os.path.join(d, s + ext)) for s in stems]) for n, d in dirs.items()}


def spreads(A, fams):
    print(f"\n{'family':>14} {'n':>3} {'internal pairwise |di-dj| /255':>32}")
    mu = {}
    for f, ms in fams.items():
        ps = [np.abs(A[a] - A[b]).mean() * 255 for i, a in enumerate(ms) for b in ms[i + 1:]]
        mu[f] = np.stack([A[m] for m in ms]).mean(0)
        print(f"{f:>14} {len(ms):3d} {np.mean(ps):32.3f}")
    ks = list(fams)
    print(f"\n{'family pair':>28} {'|meanA-meanB| /255':>20}")
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            print(f"{a+' vs '+b:>28} {np.abs(mu[a]-mu[b]).mean()*255:20.3f}")
    return mu


def loo_dev(A, ref_names):
    """deviation of each member from the mean of ALL OTHER reference members"""
    S = sum(A[n] for n in ref_names)
    out = {}
    for n in A:
        if n in ref_names:
            r = (S - A[n]) / (len(ref_names) - 1)
        else:
            r = S / len(ref_names)
        out[n] = float(np.abs(A[n] - r).mean() * 255)
    return out


# ------------------------------------------------------------------ PUBLIC (has GT)
SOLO = {"gsplatB11ut60k": 24.384, "sh3": 24.384, "m31b_nolpips": 24.379,
        "m31b_taillpips": 24.346, "gsplatB10ut8M": 24.274, "gsplatB12ut8Ms7": 24.268,
        "e17visnorm": 24.256, "e15ceil95": 24.239, "gsplatB9ut": 24.187,
        "gsplatB8pure": 24.182, "gsplatB2": 24.157, "gsplatB4warm": 24.154,
        "gsplatB1": 24.151, "gsplatB7ppisp2": 24.101, "gsplatB3": 24.046,
        "gsplatB5affine": 23.934, "e16app": 23.674, "sh2": 22.926, "sh1": 21.594,
        "sh0": 21.094, "gsplatB6bilagrid": 16.915}


def public():
    D = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
    UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
    REAL = UT4 + ["m31b_taillpips", "m31b_nolpips", "e17visnorm", "e15ceil95",
                  "gsplatB8pure", "gsplatB4warm", "gsplatB7ppisp2", "gsplatB1",
                  "gsplatB2", "gsplatB3"]
    ALL = REAL + ["e16app", "gsplatB5affine", "sh3", "sh2", "sh1", "sh0", "gsplatB6bilagrid"]
    gtd = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in ALL))[:NV]
    A = load_all({m: D(m) for m in ALL}, stems)
    print("=" * 78)
    print(f"PUBLIC HCM0181  (real test GT exists)  n={len(stems)} views")
    spreads(A, {"UT(4)": UT4, "FGS/e1x(3)": ["e15ceil95", "e16app", "e17visnorm"],
                "m31b(2)": ["m31b_nolpips", "m31b_taillpips"],
                "AA/B(4)": ["gsplatB1", "gsplatB2", "gsplatB3", "gsplatB8pure"]})
    dv = loo_dev(A, REAL)
    print(f"\n{'member':>22} {'LOOdev':>8} {'soloPSNR':>9}")
    x, y = [], []
    for m in ALL:
        print(f"{m:>22} {dv[m]:8.3f} {SOLO[m]:9.3f}")
        x.append(dv[m]); y.append(SOLO[m])
    x = np.array(x); y = np.array(y)
    r = np.corrcoef(x, y)[0, 1]
    p = np.polyfit(x, y, 1)
    # fit on the healthy members only (avoid the junk tail dominating the slope)
    xh = np.array([dv[m] for m in REAL]); yh = np.array([SOLO[m] for m in REAL])
    ph = np.polyfit(xh, yh, 1)
    rh = np.corrcoef(xh, yh)[0, 1]
    print(f"\nALL 21:   corr(LOOdev, soloPSNR) = {r:+.3f}   slope {p[0]:+.4f} dB per dev unit")
    print(f"HEALTHY 14: corr = {rh:+.3f}   slope {ph[0]:+.4f} dB per dev unit")
    return ph, rh


# ------------------------------------------------------------------ PRIVATE
def private(T, slope):
    M = {"ut7": f"/mnt/d/avv/r2r9/models/{T}_ut7/test_png",
         "ut13": f"/mnt/d/avv/r2r9/models/{T}_ut13/test_png",
         "ut42": f"/mnt/d/avv/r2r9/models/{T}_ut42/test_png",
         "ut77": f"/mnt/d/avv/r2r9/models/{T}_ut77/test_png",
         "seed101": f"/mnt/d/avv/r22_seed101/{T}/test_png",
         "mip3d": f"/mnt/d/avv/r25_mip3d/{T}/test_png",
         "r28member": f"/mnt/d/avv/r28_members/{T}/test_png",
         "champA": f"/mnt/d/avv/output_s2gates/{T}_champA/test_png",
         "memB": f"/mnt/d/avv/output_s2gates/{T}_memB/test_png",
         "memC": f"/mnt/d/avv/output_s2gates/{T}_memC/test_png"}
    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(M["ut7"]))[:NV]
    A = load_all(M, stems)
    print("=" * 78)
    print(f"PRIVATE {T}  (no GT)  n={len(stems)} views")
    spreads(A, {"UT(5)": ["ut7", "ut13", "ut42", "ut77", "seed101"],
                "mip3d/r28(2)": ["mip3d", "r28member"],
                "s2gates(3)": ["champA", "memB", "memC"]})
    REF = ["ut7", "ut13", "ut42", "ut77", "seed101", "mip3d", "r28member",
           "champA", "memB", "memC"]
    dv = loo_dev(A, REF)
    base = np.median([dv[m] for m in ("ut7", "ut13", "ut42", "ut77", "seed101")])
    print(f"\n{'member':>12} {'LOOdev':>8} {'vs UT median':>13} {'implied dPSNR (dB)':>19}")
    for m in REF:
        print(f"{m:>12} {dv[m]:8.3f} {dv[m]-base:+13.3f} {slope*(dv[m]-base):+19.3f}")
    return dv


if __name__ == "__main__":
    ph, rh = public()
    for T in ("HCM0421", "HCM0644", "HCM0539"):
        private(T, ph[0])

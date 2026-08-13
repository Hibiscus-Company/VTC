#!/usr/bin/env python
"""REFUTATION STEP A (CPU only, no GPU): place the s2gates candidate on the SAME
decorrelation/sharpness scale that HCM0181's measured members live on.

The claim reads 'dev 7.0 vs in-pool 3.15 => 2.2x more decorrelated => worth more'.
Decorrelation is ambiguous: a member deviates a lot either because its errors are
independent (good) or because it is simply a worse/softer model (bad).  On HCM0181 we
have REAL GT and MEASURED per-member ensemble value, so we can calibrate what a 2.2x
deviation ratio actually buys.
"""
import os, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
NV = int(os.environ.get("NV", "12"))


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def grad(a):
    g = a.mean(2)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2


def hf(a):
    """energy above the 2x downsample band -- 'softness' proxy"""
    g = a.mean(2)
    h, w = g.shape
    d = g[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    up = np.repeat(np.repeat(d, 2, 0), 2, 1)
    return float(((g[: h // 2 * 2, : w // 2 * 2] - up) ** 2).mean() ** 0.5)


def report(tag, ensdirs, ensw, memdirs, stems, ext=".png"):
    ensw = np.asarray(ensw, float); ensw = ensw / ensw.sum()
    names = list(memdirs)
    dev = {n: 0.0 for n in names}
    sh = {n: 0.0 for n in names}
    ens_sh = 0.0
    for s in stems:
        E = None
        for d, w in zip(ensdirs, ensw):
            a = ld(os.path.join(d, s + ext))
            E = a * w if E is None else E + a * w
        ens_sh += hf(E)
        for n in names:
            a = ld(os.path.join(memdirs[n], s + ext))
            dev[n] += np.abs(a - E).mean() * 255.0
            sh[n] += hf(a)
    N = len(stems)
    print(f"\n### {tag}   n={N} views")
    print(f"{'member':>26} {'dev/255':>9} {'HFrms':>8} {'HF/ens':>8}")
    for n in names:
        print(f"{n:>26} {dev[n]/N:9.3f} {sh[n]/N:8.5f} {sh[n]/ens_sh:8.3f}")
    print(f"{'[ensemble mean]':>26} {0.0:9.3f} {ens_sh/N:8.5f} {1.0:8.3f}")
    return {n: dev[n] / N for n in names}, {n: sh[n] / N for n in names}, ens_sh / N


# ---------------------------------------------------------------- PRIVATE towers
def private(T):
    R22 = f"/mnt/d/avv/r22/tower_ens/{T}/png_ens"
    MIP = f"/mnt/d/avv/r25_mip3d/{T}/test_png"
    NEW = f"/mnt/d/avv/r28_members/{T}/test_png"
    mem = {
        "ut7": f"/mnt/d/avv/r2r9/models/{T}_ut7/test_png",
        "ut13": f"/mnt/d/avv/r2r9/models/{T}_ut13/test_png",
        "ut42": f"/mnt/d/avv/r2r9/models/{T}_ut42/test_png",
        "ut77": f"/mnt/d/avv/r2r9/models/{T}_ut77/test_png",
        "seed101": f"/mnt/d/avv/r22_seed101/{T}/test_png",
        "mip3d(in pool)": MIP,
        "r28member(in pool)": NEW,
        "[r22 6-mean](in pool)": R22,
        "s2gates champA": f"/mnt/d/avv/output_s2gates/{T}_champA/test_png",
        "s2gates memB": f"/mnt/d/avv/output_s2gates/{T}_memB/test_png",
        "s2gates memC": f"/mnt/d/avv/output_s2gates/{T}_memC/test_png",
    }
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(R22))[:NV]
    return report(f"PRIVATE {T} (r29 shipped 8-member mean = 4:1:1)",
                  [R22, MIP, NEW], [4, 1, 1], mem, stems)


# ---------------------------------------------------------------- PUBLIC HCM0181
def public():
    D = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
    UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
    others = ["m31b_taillpips", "m31b_nolpips", "e17visnorm", "e15ceil95", "e16app",
              "gsplatB8pure", "gsplatB4warm", "gsplatB7ppisp2", "gsplatB1", "gsplatB2",
              "gsplatB3", "gsplatB5affine", "sh3", "sh2", "sh1", "sh0", "gsplatB6bilagrid"]
    mem = {m + " (in pool)": D(m) for m in UT4}
    mem.update({m: D(m) for m in others})
    gtd = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png"))
                          for m in UT4 + others))[:NV]
    d, s, e = report("PUBLIC HCM0181 (4-member UT pool, uniform)",
                     [D(m) for m in UT4], [1, 1, 1, 1], mem, stems)
    # GT-anchored: per-member solo PSNR + measured k=4 ensemble value (div_fixk)
    val = {"e17visnorm": +0.3064, "e15ceil95": +0.2973, "gsplatB8pure": +0.2381,
           "e16app": +0.2104, "gsplatB4warm": +0.1805, "m31b_taillpips": +0.1662,
           "gsplatB7ppisp2": +0.1527, "m31b_nolpips": +0.0670, "gsplatB1": +0.0481,
           "gsplatB2": +0.0464}
    inpool = np.median([d[m + " (in pool)"] for m in UT4])
    print(f"\nHCM0181 in-pool UT median dev = {inpool:.3f}")
    print(f"{'member':>22} {'dev':>7} {'ratio':>7} {'HF/ens':>7} {'measured k4 value':>18}")
    rows = []
    for m in others:
        r = d[m] / inpool
        v = val.get(m)
        vs = f"{v:+18.4f}" if v is not None else f"{'(excluded as junk)':>18}"
        print(f"{m:>22} {d[m]:7.3f} {r:7.2f} {s[m]/e:7.3f} {vs}")
        if v is not None:
            rows.append((r, v))
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    print(f"\ncorr(dev ratio, measured ensemble value) = {np.corrcoef(x, y)[0,1]:+.3f}"
          f"   [claim needs this POSITIVE and strong]")
    return d, s, e, inpool


if __name__ == "__main__":
    pd_, ps_, pe_ = {}, {}, {}
    for T in ("HCM0421", "HCM0644"):
        a, b, c = private(T)
        inpool = np.median([a[k] for k in ("ut7", "ut13", "ut42", "ut77", "seed101")])
        print(f"  -> {T}: in-pool UT median dev {inpool:.3f}; "
              f"champA {a['s2gates champA']:.3f} (ratio {a['s2gates champA']/inpool:.2f}); "
              f"champA HF/ens {b['s2gates champA']/c:.3f}")
    public()

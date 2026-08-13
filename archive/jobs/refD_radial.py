#!/usr/bin/env python
"""REFUTATION STEP D: is the s2gates deviation a RADIAL (distortion-model) residual?

The set2 towers are SIMPLE_RADIAL, k1 ~ +0.009. The UT pool renders natively in DISTORTED space.
The s2gates members are FastGS/3DGS-lineage runs whose cfg_args show source images='images' with
a pinhole projection -- if the k1 term is absorbed into geometry rather than modelled, the
member's disagreement with the pool should grow with image RADIUS, which a pure "independent
model noise" member would not do.

A radially-structured deviation is the one kind of disagreement that pixel-mean ensembling
CANNOT cancel: it is a shared, deterministic bias, and it lands hardest exactly where the lens
field (fit on the pool's own renders) does not expect it.
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
NV = int(os.environ.get("NV", "8"))
ld = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
NB = 6


def radial_profile(dirs_ens, w, cands, stems):
    w = np.asarray(w, float); w /= w.sum()
    prof, cnt = {c: np.zeros(NB) for c in cands}, np.zeros(NB)
    for s in stems:
        E = None
        for d, ww in zip(dirs_ens, w):
            a = ld(os.path.join(d, s + ".png"))
            E = a * ww if E is None else E + a * ww
        H, W, _ = E.shape
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((yy - H / 2) / (H / 2)) ** 2 + ((xx - W / 2) / (W / 2)) ** 2)
        r = r / r.max()
        b = np.clip((r * NB).astype(int), 0, NB - 1)
        if cnt.sum() == 0:
            cnt = np.bincount(b.ravel(), minlength=NB).astype(float)
        for c, d in cands.items():
            dv = np.abs(ld(os.path.join(d, s + ".png")) - E).mean(2) * 255
            prof[c] += np.bincount(b.ravel(), weights=dv.ravel(), minlength=NB)
    return {c: prof[c] / (cnt * len(stems)) for c in cands}


for T in ("HCM0421", "HCM0644"):
    R22 = f"/mnt/d/avv/r22/tower_ens/{T}/png_ens"
    cands = {"ut7": f"/mnt/d/avv/r2r9/models/{T}_ut7/test_png",
             "ut42": f"/mnt/d/avv/r2r9/models/{T}_ut42/test_png",
             "seed101": f"/mnt/d/avv/r22_seed101/{T}/test_png",
             "mip3d": f"/mnt/d/avv/r25_mip3d/{T}/test_png",
             "champA": f"/mnt/d/avv/output_s2gates/{T}_champA/test_png",
             "memB": f"/mnt/d/avv/output_s2gates/{T}_memB/test_png"}
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(R22))[:NV]
    p = radial_profile([R22, f"/mnt/d/avv/r25_mip3d/{T}/test_png",
                        f"/mnt/d/avv/r28_members/{T}/test_png"], [4, 1, 1], cands, stems)
    print(f"\n### {T}: mean |member - pool mean| /255 by normalized image radius (n={len(stems)})")
    print(f"{'member':>9} " + " ".join(f"r{i}".rjust(7) for i in range(NB)) + "   edge/centre")
    for c in cands:
        v = p[c]
        print(f"{c:>9} " + " ".join(f"{x:7.3f}" for x in v)
              + f"   {v[-1]/v[0]:11.2f}")

# public analogue
D = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
cands = {m: D(m) for m in UT4 + ["e17visnorm", "e15ceil95", "e16app", "gsplatB8pure"]}
gtd = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
stems = sorted(os.path.splitext(f)[0] for f in os.listdir(gtd)
               if os.path.exists(os.path.join(D("e16app"), os.path.splitext(f)[0] + ".png")))[:NV]
p = radial_profile([D(m) for m in UT4], [1, 1, 1, 1], cands, stems)
print(f"\n### PUBLIC HCM0181 (UT4 pool mean), n={len(stems)}")
print(f"{'member':>16} " + " ".join(f"r{i}".rjust(7) for i in range(NB)) + "   edge/centre")
for c in cands:
    v = p[c]
    print(f"{c:>16} " + " ".join(f"{x:7.3f}" for x in v) + f"   {v[-1]/v[0]:11.2f}")

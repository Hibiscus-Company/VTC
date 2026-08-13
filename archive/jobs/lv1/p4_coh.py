#!/usr/bin/env python
"""PHASE C (diagnostic, CPU only): WHY a level-1 boost cannot help.

Per Laplacian level l, against the real HCM0181 test GT, for the k-member mean:
  deficit_l = sqrt(E(L_gt)/E(L_mean))                 amplitude the mean is "missing"
  g*_l      = <L_mean, L_gt> / <L_mean, L_mean>       MSE-OPTIMAL scalar gain on that band
  coh_l     = <L_mean, L_gt> / sqrt(E_mean * E_gt)    correlation of the band with GT
A band can only be usefully re-energised if g*_l > 1.  If g*_l <= 1 while deficit_l > 1 the
missing energy is INCOHERENT with GT and any boost is pure added noise.
Also reports g* under the ADDITIVE operator form actually shipped (boost direction = the r map),
i.e. the MSE-optimal lambda for  L + lam*(r-1)*L  at each level.
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
sys.path.insert(0, JOB)
from PIL import Image
from lapfuse import lap_pyr, boxf, _K
Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(2)

ROOT = "/mnt/d/avv/output"
POOL8 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
         "m31b_nolpips", "e17visnorm", "gsplatB8pure", "gsplatB4warm"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
NLEV, WIN, CLAMP = 5, 3, 4.0


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    K = int(os.environ.get("K", "8"))
    NIMG = int(os.environ.get("NIMG", "20"))
    dirs = [os.path.join(ROOT, "HCM0181_" + v, "test_poses_renders_png") for v in POOL8[:K]]
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))[:NIMG]
    print(f"k={K} n={len(stems)}", flush=True)

    Emean = np.zeros(NLEV); Egt = np.zeros(NLEV); Cross = np.zeros(NLEV)
    # additive-operator statistics:  d = (r-1)*L   ->  lam* = <d, L_gt - L_mean> / <d,d>
    Add_dd = np.zeros(NLEV); Add_dg = np.zeros(NLEV)
    for c, st in enumerate(stems):
        mlaps = [lap_pyr(load(os.path.join(d, st + ".png")), NLEV, _K)[0] for d in dirs]
        gl = lap_pyr(load(os.path.join(GTD, gt_by[st])), NLEV, _K)[0]
        for l in range(NLEV):
            stk = torch.cat([m[l] for m in mlaps], 0)          # [k,3,h,w]
            mu = stk.mean(0, keepdim=True)                       # [1,3,h,w]
            Eb = boxf((mu ** 2).sum(1, keepdim=True), WIN)
            V = boxf(((stk - mu) ** 2).sum(1, keepdim=True), WIN).mean(0, keepdim=True) \
                * (K / (K - 1.0))
            r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=CLAMP)
            d = (r - 1.0) * mu
            Emean[l] += float((mu ** 2).sum())
            Egt[l] += float((gl[l] ** 2).sum())
            Cross[l] += float((mu * gl[l]).sum())
            Add_dd[l] += float((d ** 2).sum())
            Add_dg[l] += float((d * (gl[l] - mu)).sum())
            del stk, mu, Eb, V, r, d
        del mlaps, gl
        if c % 5 == 0:
            print(f"  {c}/{len(stems)}", flush=True)

    print(f"\nHCM0181 real test GT, k={K}, n={len(stems)}, raw renders")
    print(f"{'lev':>3} {'deficit':>8} {'g* scale':>9} {'coh':>7} {'lam* add':>9} "
          f"{'maxgain@lam*':>13}")
    out = []
    for l in range(NLEV):
        defi = np.sqrt(Egt[l] / Emean[l])
        g = Cross[l] / Emean[l]
        coh = Cross[l] / np.sqrt(Emean[l] * Egt[l])
        lam = Add_dg[l] / max(Add_dd[l], 1e-12)
        gain = Add_dg[l] ** 2 / max(Add_dd[l], 1e-12)     # MSE reduction at optimal lam
        print(f"{l:>3} {defi:8.4f} {g:9.4f} {coh:7.4f} {lam:9.4f} {gain:13.4g}")
        out.append(dict(lev=l, deficit=float(defi), g_star=float(g), coh=float(coh),
                        lam_star_additive=float(lam), mse_drop=float(gain)))
    json.dump(out, open(os.path.join(HERE, f"p4_coh_k{K}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

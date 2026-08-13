#!/usr/bin/env python
"""INDEPENDENT REFUTATION of the winsor-nudge combiner claim.

Written from scratch against the PRODUCTION operator (energy_restore.restore) and the
production field code, not against cb_run.py.  Scores through the full shipped chain at
the ACTUAL r30 tower shape (k=10 uniform, gauss1 field x1.30, JPEG q100/ss2) instead of
r29's retired 4:1:1.

Arms:
  m10           control: plain 10-member mean, lam=1.0
  wx10_t0.5     the claimed operator at the CORRECT k (=10)
  wx10_k7dose   the LITERAL shipped command's dose (t/k with k=7 while the ens is k=10)
  wx10_sub7     the LITERAL shipped OPERATOR: order statistics over a 7-member SUBSET
  anti10        sign placebo: -t
  m10_lam1.10   knob-retune control: is this just an under-tuned energy-restore lambda?
  mA8 / wxA8    replication of the claim's own headline config (PoolA, k=8)
"""
import io, os, sys, time, json
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore                      # PRODUCTION operator
from fieldlib import LooPool, gauss_smooth, upsample, warp
from lapfuse import lap_pyr, _K

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(2)
SHIP = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG, GAIN = "HCM0181", 1.30
DEV = os.environ.get("REFW_DEV", "cuda:1")
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"

# k=10 argmax pool = the r30 production tower shape (kcurve2.py ORDER[:10])
P10 = ["gsplatB11ut60k", "sh3", "m31b_nolpips", "m31b_taillpips", "gsplatB10ut8M",
       "gsplatB12ut8Ms7", "e17visnorm", "e15ceil95", "gsplatB9ut", "gsplatB8pure"]
# the claim's PoolA (k=8) -- a strict subset of P10
PA8 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
       "m31b_nolpips", "m31b_taillpips", "e15ceil95", "e17visnorm"]
IA8 = [P10.index(m) for m in PA8]
# a 7-of-10 subset, mimicking production passing 7 member dirs for a 10-member ensemble
SUB7 = [0, 2, 4, 5, 6, 7, 8]


def r8(t):
    return (t.clamp(0, 1) * 255.0 + 0.5).floor().clamp(0, 255) / 255.0


def gap(X):
    """(x_(2)-x_(1)) + (x_(k-1)-x_(k)) ; corr = (t/k)*gap  == t*(winsor1 - mean)*k/k"""
    S, _ = torch.sort(X, dim=0)
    k = X.shape[0]
    return (S[1] - S[0]) + (S[k - 2] - S[k - 1])


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(DEV).eval()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in P10))
    c = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in c["HW"]]
    lens = upsample(gauss_smooth(LooPool(c["s8"]).pooled("median"), 1), H, W, "cubic") * GAIN
    K = _K.to(DEV)

    ARMS = ["m10", "wx10_t0.5", "wx10_k7dose", "wx10_sub7", "anti10", "m10_lam1.10",
            "mA8", "wxA8_t0.5"]
    per = {a: [] for a in ARMS}
    nby = {a: 0 for a in ARMS}
    t0 = time.time()
    for n, s in enumerate(stems):
        X = torch.stack([torch.from_numpy(
            np.asarray(Image.open(os.path.join(D(m), s + ".png")).convert("RGB"),
                       np.float32) / 255.0).permute(2, 0, 1) for m in P10], 0).to(DEV)
        mem10 = [X[i:i + 1] for i in range(10)]
        mem8 = [X[i:i + 1] for i in IA8]
        m10 = r8(X.mean(0))
        m8 = r8(X[IA8].mean(0))
        g10 = gap(X)
        g8 = gap(X[IA8])
        g7 = gap(X[SUB7])
        gt = torch.from_numpy(np.asarray(Image.open(os.path.join(GTD, gt_by[s]))
                                         .convert("RGB"), np.float32) / 255.0
                              ).permute(2, 0, 1).unsqueeze(0).to(DEV)

        build = {
            "m10":         (m10, mem10, 10, 1.0),
            "wx10_t0.5":   (r8(m10 + (0.5 / 10) * g10), mem10, 10, 1.0),
            "wx10_k7dose": (r8(m10 + (0.5 / 7) * g10), mem10, 10, 1.0),
            "wx10_sub7":   (r8(m10 + (0.5 / 7) * g7), mem10, 10, 1.0),
            "anti10":      (r8(m10 - (0.5 / 10) * g10), mem10, 10, 1.0),
            "m10_lam1.10": (m10, mem10, 10, 1.10),
            "mA8":         (m8, mem8, 8, 1.0),
            "wxA8_t0.5":   (r8(m8 + (0.5 / 8) * g8), mem8, 8, 1.0),
        }
        for a in ARMS:
            e, mem, k, lam = build[a]
            o = restore(e.unsqueeze(0), mem, lam, k).clamp(0, 1)
            x = o[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(warp(np.ascontiguousarray(x), lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIP)
            nby[a] += b.tell()
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(float(((r - gt) ** 2).mean()), 1e-12))
                S = float(repo_ssim(r, gt))
                L = float(vgg(r * 2 - 1, gt * 2 - 1).item())
            per[a].append((P, S, L, 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.))))
        del X, mem10, mem8, gt
        if n % 10 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    A = {a: np.array(per[a]) for a in ARMS}
    json.dump({a: A[a].tolist() for a in ARMS}, open(f"{HERE}/refW.json", "w"))
    print(f"\nREFUTE-W  {TAG}  n={N}  gauss1 field x{GAIN}  JPEG q100/ss2  PRODUCTION restore")
    print(f"{'arm':>14} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'MB':>7}")
    for a in ARMS:
        m = A[a].mean(0)
        print(f"{a:>14} {m[3]:9.4f} {m[0]:8.4f} {m[1]:8.5f} {m[2]:8.5f} {nby[a]/1e6:7.2f}")

    def pr(a, b):
        d = A[a][:, 3] - A[b][:, 3]
        se = d.std(ddof=1) / np.sqrt(N)
        print(f"{a+'  -  '+b:>38} {d.mean():+9.4f} {se:7.4f} {d.mean()/se:+7.2f} "
              f"{int((d>0).sum()):3d}/{N}")

    print(f"\n{'contrast':>38} {'delta':>9} {'se':>7} {'t':>7} {'wins':>7}")
    for a, b in [("wx10_t0.5", "m10"), ("wx10_k7dose", "m10"), ("wx10_sub7", "m10"),
                 ("anti10", "m10"), ("m10_lam1.10", "m10"), ("wxA8_t0.5", "mA8"),
                 ("m10", "mA8")]:
        pr(a, b)


if __name__ == "__main__":
    main()

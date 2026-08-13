"""RESAMPLING TONE SPACE.  The lanczos remap runs on sRGB code values (p=1).  Moving it to
LINEAR light lost -0.143/scene on 5/5 towers (lens_new.py).  That is a steep gradient, so the
axis is real -- ask whether the optimum is on the other side of p=1, i.e. in a MORE compressive
space.  Warp in x**p space (p<1 more compressive than sRGB code, p>1 towards linear).
Pure resampling-kernel class: PER-IMAGE, no energy added, same taps re-weighted.
"""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from fieldlib import LooPool, gauss_smooth, upsample, warp
Image.MAX_IMAGE_PIXELS = None
GAIN = 1.30
SHIP = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAGS = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
PS = [0.40, 0.70, 1.00, 1.40]
NMAX = int(os.environ.get("NMAX", "30"))
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev = "cuda" if torch.cuda.is_available() else "cpu"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
ldf = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
res = {}
for TAG in TAGS:
    D = f"/mnt/d/avv/output/{TAG}_gsplatB9ut/test_poses_renders_png"
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by if os.path.exists(os.path.join(D, s + ".png")))[:NMAX]
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H, W = [int(x) for x in cache["HW"]]
    FU = upsample(gauss_smooth(LooPool(cache["s8"]).pooled("median"), 1), H, W, "cubic") * GAIN
    acc = {p: [0., 0., 0.] for p in PS}; per = {p: [] for p in PS}; t0 = time.time()
    for n, s in enumerate(stems):
        o = ldf(os.path.join(D, s + ".png"))
        g = torch.from_numpy(ldf(os.path.join(gtd, gt_by[s]))).permute(2, 0, 1).unsqueeze(0).to(dev)
        for p in PS:
            x = np.ascontiguousarray(o ** p) if p != 1.0 else o
            x = np.clip(warp(x, FU, "lanczos"), 0, 1)
            if p != 1.0:
                x = x ** (1.0 / p)
            b = io.BytesIO()
            Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIP)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1. / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g)); L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[p][0] += P; acc[p][1] += S; acc[p][2] += L
            per[p].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.)))
        if n % 15 == 0: print(f"  {TAG} {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    N = len(stems)
    print(f"\n== {TAG} n={N}")
    for p in PS:
        P, S, L = (v / N for v in acc[p])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.))
        d = np.array(per[p]) - np.array(per[1.00])
        print(f"  p={p:<5} {sc:9.4f} PSNR {P:7.4f} SSIM {S:.5f} LPIPS {L:.5f} "
              f"d {d.mean():+8.4f} se {d.std(ddof=1)/np.sqrt(N):6.4f} wins {int((d>0).sum()):3d}/{N}")
        res.setdefault(p, {})[TAG] = d.mean()
    sys.stdout.flush()
print("\n===== SUMMARY d vs p=1 (sRGB code space, shipped) =====")
for p in PS:
    v = list(res[p].values())
    print(f"p={p:<5} MEAN {np.mean(v):+8.4f}  scenes+ {sum(1 for x in v if x>0)}/{len(v)}  "
          + " ".join(f"{k}{x:+.4f}" for k, x in res[p].items()))

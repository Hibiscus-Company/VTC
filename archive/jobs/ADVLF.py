"""ADVERSARIAL REPRODUCTION of the 'low-frequency residual +0.455 oracle' claim.

Runs the FULL SHIPPED CHAIN on the public production harness (HCM0181, 8 members, real test GT):
    ensemble mean -> restore(lam) -> median lens field lanczos4 (gain 1.30) -> JPEG q100/ss2
then applies, POST-CHAIN, the arms that matter:

  base        the shipped output
  oracleLF    base + Gauss(GT-base, 12.8)        <- the claim's operator, needs GT (upper bound)
  oracleDC    base + per-channel mean(GT-base)   <- global exposure oracle (per image)
  oracleAFF   per-channel least-squares a*x+b vs GT   <- global affine oracle (per image)
  pooledLOO   base + LOO-median of the LF residual over the OTHER test views
              <- the only GT-free-at-test form of a spatially varying LF correction:
                 a static per-scene photometric field, twin of the shipped lens field
  pooledLOOm  same with the mean instead of the median

Also reports the view-consistent vs view-specific energy split of the LF residual.
"""
import os, sys, io, json
import numpy as np, torch, cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/lens")
from utils.loss_utils import ssim as repo_ssim
from energy_restore import restore, SHIPPED_JPEG
from lens.fieldlib import LooPool, upsample, warp
import lpips as lpips_pkg

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(6)
dev = os.environ.get("DEV", "cuda:1")
TAG = "HCM0181"; ROOT = "/mnt/d/avv/output"; GAIN = 1.30
LAM = float(os.environ.get("LAM", "0.5"))
NIMG = int(os.environ.get("NIMG", "24"))
SIG = 12.8
ORDER = ["gsplatB11ut60k", "sh3", "m31b_nolpips", "m31b_taillpips", "gsplatB10ut8M",
         "gsplatB12ut8Ms7", "e17visnorm", "e15ceil95"]
D = lambda m: f"{ROOT}/{TAG}_{m}/test_poses_renders_png"
gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
gt = {os.path.splitext(f)[0]: f"{gtd}/{f}" for f in os.listdir(gtd)}
vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.) \
    .permute(2, 0, 1).unsqueeze(0)
sc = lambda p, s, l: 100 * (0.4 * (1 - l) + 0.3 * s + 0.3 * p / 50)

stems = sorted(s for s in gt if all(os.path.exists(f"{D(m)}/{s}.png") for m in ORDER))
stems = stems[::max(1, len(stems) // NIMG)][:NIMG]
cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz"); H, W = [int(x) for x in cache["HW"]]
lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic") * GAIN
K = len(ORDER)
print(f"{TAG} n={len(stems)} k={K} lam={LAM} sigma={SIG} dev={dev}", flush=True)


def jenc(t):
    a = (t.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", **SHIPPED_JPEG)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.


# ---- pass 1: build shipped outputs + GT, collect LF residuals at ds8 ----
outs, gts, Rs = [], [], []
for i, s in enumerate(stems):
    mem = [ld(f"{D(m)}/{s}.png") for m in ORDER]
    fm = torch.stack(mem).mean(0)
    t = restore(fm.to(dev), [m.to(dev) for m in mem], LAM, K).clamp(0, 1).cpu()
    hw = t[0].permute(1, 2, 0).numpy().astype(np.float32)
    o = jenc(torch.from_numpy(warp(hw, lens, "lanczos")).permute(2, 0, 1).unsqueeze(0))  # HxWx3 float
    g = np.asarray(Image.open(gt[s]).convert("RGB"), dtype=np.float32) / 255.
    R = cv2.GaussianBlur(g - o, (0, 0), SIG, borderType=cv2.BORDER_REFLECT)
    outs.append(o); gts.append(g)
    Rs.append(cv2.resize(R, (R.shape[1] // 8, R.shape[0] // 8), interpolation=cv2.INTER_AREA))
    if i % 6 == 0: print("  img", i, flush=True)
Rs = np.stack(Rs); N = len(stems)
Et = float((Rs ** 2).mean()); Md = np.median(Rs, 0); Mn = Rs.mean(0)
print(f"\nLF residual (sigma {SIG}) RMS {255*np.sqrt(Et):.3f} LSB | pooled-median RMS "
      f"{255*np.sqrt((Md**2).mean()):.3f} | after median removal {255*np.sqrt(((Rs-Md)**2).mean()):.3f}")
print(f"VIEW-CONSISTENT energy fraction: median {100*(1-((Rs-Md)**2).mean()/Et):.2f}%  "
      f"mean {100*(1-((Rs-Mn)**2).mean()/Et):.2f}%", flush=True)

ARMS = ["base", "oracleLF", "oracleDC", "oracleAFF", "pooledLOO", "pooledLOOm"]
acc = {a: np.zeros(3) for a in ARMS}
per = {a: [] for a in ARMS}
with torch.no_grad():
    for i in range(N):
        o, g = outs[i], gts[i]
        gt_t = torch.from_numpy(np.ascontiguousarray(g)).permute(2, 0, 1).unsqueeze(0).to(dev)
        v = {"base": o, "oracleLF": o + cv2.GaussianBlur(g - o, (0, 0), SIG,
                                                         borderType=cv2.BORDER_REFLECT)}
        v["oracleDC"] = o + (g - o).mean((0, 1))[None, None, :]
        aff = o.copy()
        for c in range(3):
            x, y = o[..., c].ravel(), g[..., c].ravel()
            A = np.polyfit(x, y, 1); aff[..., c] = A[0] * o[..., c] + A[1]
        v["oracleAFF"] = aff
        for tag, P in (("pooledLOO", np.median(np.delete(Rs, i, 0), 0)),
                       ("pooledLOOm", np.delete(Rs, i, 0).mean(0))):
            v[tag] = o + cv2.resize(P, (W, H), interpolation=cv2.INTER_CUBIC)
        for a in ARMS:
            q = np.clip(v[a], 0, 1)
            q = np.round(q * 255).astype(np.uint8).astype(np.float32) / 255.   # 8-bit deliverable
            t = torch.from_numpy(np.ascontiguousarray(q)).permute(2, 0, 1).unsqueeze(0).to(dev)
            m = [10 * np.log10(1 / max(((t - gt_t) ** 2).mean().item(), 1e-12)),
                 float(repo_ssim(t, gt_t)), float(vgg(t * 2 - 1, gt_t * 2 - 1).mean())]
            acc[a] += m; per[a].append(sc(*m))

print(f"\n=== FULL SHIPPED CHAIN, {TAG}, n={N} real test poses, lam={LAM} ===")
print(f"{'arm':>11} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'SCORE':>9} {'dSCORE':>9} {'dLPIPS':>9} {'win/N':>7}")
b = acc["base"] / N; bs = sc(*b)
for a in ARMS:
    m = acc[a] / N; s = sc(*m)
    w = sum(1 for k in range(N) if per[a][k] > per["base"][k])
    print(f"{a:>11} {m[0]:8.4f} {m[1]:8.5f} {m[2]:8.5f} {s:9.4f} {s-bs:+9.4f} {m[2]-b[2]:+9.5f} {w:4d}/{N}")
d = np.array(per["pooledLOO"]) - np.array(per["base"])
print(f"pooledLOO per-image dScore mean {d.mean():+.4f} sd {d.std(ddof=1):.4f} "
      f"sem {d.std(ddof=1)/np.sqrt(N):.4f}")
print("DONE")

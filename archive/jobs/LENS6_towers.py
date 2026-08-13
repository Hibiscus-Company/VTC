"""LENS-6: is the 'private towers all ~78.6, within 0.1 of ceiling' anchor real?
(A) per-tower SOLO submetrics for the 5 public towers, full shipped chain, real test GT
(B) radial power ratio render/GT (public, real test GT) and render/OWN-TRAIN-PHOTOS (all 10 towers)
    -> train-photo proxy is a LEGAL surface computable for the private towers too."""
import io, os, sys, time, json
import numpy as np, torch
from PIL import Image
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from fieldlib import LooPool, gauss_smooth, upsample, warp
Image.MAX_IMAGE_PIXELS = None
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev = "cuda" if torch.cuda.is_available() else "cpu"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

PUB = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
PRV = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]

# ---------- radial power helper ----------
_win = {}
def rad_power(path, nmax=60):
    fs = sorted(os.listdir(path))
    step = max(1, len(fs) // nmax)
    fs = fs[::step][:nmax]
    acc = None; n = 0
    for f in fs:
        a = np.asarray(Image.open(os.path.join(path, f)).convert("L"), dtype=np.float32) / 255.
        H, W = a.shape
        if (H, W) not in _win:
            _win[(H, W)] = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
        a = (a - a.mean()) * _win[(H, W)]
        P = np.abs(np.fft.rfft2(a)) ** 2
        if acc is None:
            acc = np.zeros_like(P)
            fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.rfftfreq(W)[None, :]
            rr = np.sqrt(fy ** 2 + fx ** 2)
        acc += P; n += 1
    acc /= n
    bands = {}
    for nm, lo, hi in [("low", 0.02, 0.08), ("mid", 0.10, 0.25), ("high", 0.25, 0.45)]:
        m = (rr >= lo) & (rr < hi)
        bands[nm] = float(acc[m].mean())
    return bands, n

# ---------- (A) public solo submetrics through the shipped chain ----------
out = {"solo": {}, "pow": {}}
for TAG in PUB:
    RD = f"/mnt/d/avv/output/{TAG}_gsplatB9ut/test_poses_renders_png"
    GD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    cp = f"{HERE}/lens/cache/pub_{TAG}.npz"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GD)}
    stems = sorted(s for s in gt_by if os.path.exists(os.path.join(RD, s + ".png")))
    cache = np.load(cp); H, W = [int(x) for x in cache["HW"]]
    FU = upsample(gauss_smooth(LooPool(cache["s8"]).pooled("median"), 1), H, W, "cubic") * 1.30
    P = S = L = 0.0; per = []
    for s in stems:
        img = np.asarray(Image.open(os.path.join(RD, s + ".png")).convert("RGB"), dtype=np.float32) / 255.
        u8 = (np.clip(warp(img, FU, "lanczos"), 0, 1) * 255 + 0.5).astype(np.uint8)
        b = io.BytesIO(); Image.fromarray(u8).save(b, "JPEG", quality=100, subsampling=2, optimize=True, progressive=True)
        j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.
        r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(GD, gt_by[s])).convert("RGB"), dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            p = 10 * np.log10(1. / max(((r - g) ** 2).mean().item(), 1e-12)); ss = float(repo_ssim(r, g)); ll = float(vgg(r * 2 - 1, g * 2 - 1).item())
        P += p; S += ss; L += ll
        per.append(100 * (0.4 * (1 - ll) + 0.3 * ss + 0.3 * min(p / 50., 1.)))
    N = len(stems); P /= N; S /= N; L /= N
    out["solo"][TAG] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.)),
                            psnr=P, ssim=S, lpips=L, n=N,
                            se=float(np.std(per, ddof=1) / np.sqrt(N)))
    print(f"  solo {TAG}: {out['solo'][TAG]['score']:.4f}  PSNR {P:.3f} SSIM {S:.5f} LPIPS {L:.5f}  se {out['solo'][TAG]['se']:.3f}", flush=True)

# ---------- (B) power ratios ----------
for TAG in PUB:
    rb, nr = rad_power(f"/mnt/d/avv/output/{TAG}_gsplatB9ut/test_poses_renders_png")
    gb, ng = rad_power(f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images")
    tb, nt = rad_power(f"/mnt/d/avv/data/phase1/public_set/{TAG}/train/images")
    out["pow"][TAG] = dict(set="pub", r=rb, gt=gb, tr=tb,
                           ratio_gt={k: rb[k] / gb[k] for k in rb},
                           ratio_tr={k: rb[k] / tb[k] for k in rb}, n=(nr, ng, nt))
    print(f"  pow {TAG}: vs TESTGT mid {rb['mid']/gb['mid']:.3f} high {rb['high']/gb['high']:.3f} | vs TRAIN mid {rb['mid']/tb['mid']:.3f} high {rb['high']/tb['high']:.3f}", flush=True)
for TAG in PRV:
    rb, nr = rad_power(f"/mnt/d/avv/output_s2gates/{TAG}_champA/test_png")
    tb, nt = rad_power(f"/mnt/d/avv/data/phase1/private_set2/{TAG}/train/images")
    out["pow"][TAG] = dict(set="prv", r=rb, tr=tb, ratio_tr={k: rb[k] / tb[k] for k in rb}, n=(nr, nt))
    print(f"  pow {TAG}: vs TRAIN mid {rb['mid']/tb['mid']:.3f} high {rb['high']/tb['high']:.3f}", flush=True)
json.dump(out, open(f"{HERE}/LENS6_towers.json", "w"), indent=1)
print("WROTE", f"{HERE}/LENS6_towers.json")

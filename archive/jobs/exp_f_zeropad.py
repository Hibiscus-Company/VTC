"""(a-2) How much does the repo SSIM's ZERO-PADDING change our score, and how much of
the SSIM number is actually *decided* by the outer 5-px ring?"""
import sys
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from harness import *
import torch.nn.functional as Fn

W_ = 11; PAD = 5


def ssim_pad(img1, img2, mode):
    """mode: 'zeros' (= repo), 'reflect', 'replicate', or 'valid' (crop 5px, no padding)."""
    C = img1.size(-3)
    win = create_window(W_, C).type_as(img1).to(img1.device)
    if mode == "valid":
        f = lambda x: Fn.conv2d(x, win, groups=C)
    elif mode == "zeros":
        f = lambda x: Fn.conv2d(x, win, padding=PAD, groups=C)
    else:
        f = lambda x: Fn.conv2d(Fn.pad(x, (PAD,) * 4, mode=mode), win, groups=C)
    mu1, mu2 = f(img1), f(img2)
    m1s, m2s, m12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = f(img1 * img1) - m1s; s2 = f(img2 * img2) - m2s; s12 = f(img1 * img2) - m12
    C1, C2 = 1e-4, 9e-4
    return ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1s + m2s + C1) * (s1 + s2 + C2))


def dm(H, W):
    yy = torch.arange(H, device=DEV).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=DEV).view(1, W).expand(H, W)
    return torch.minimum(torch.minimum(yy, H - 1 - yy), torch.minimum(xx, W - 1 - xx))


SETS = [("HCM0181/k4", K4, GT_TEST.format(s="HCM0181"))] + \
       [(f"{s}/B9ut", R_TEST.format(s=s), GT_TEST.format(s=s)) for s in SCENES]

print("A) padding-mode sensitivity of the repo SSIM (what the zero-pad is worth to us)")
print(f"{'set':14s} {'zeros(repo)':>12s} {'reflect':>10s} {'replicate':>10s} {'valid-crop':>11s}"
      f" {'zeros-valid (pts)':>18s}")
for tag, rd, gd in SETS:
    acc = {m: 0.0 for m in ("zeros", "reflect", "replicate", "valid")}
    n = 0
    with torch.no_grad():
        for _, rp, gp in pairs(rd, gd):
            r = load(rp).to(DEV); g = load(gp).to(DEV)
            for m in acc: acc[m] += float(ssim_pad(r, g, m).mean())
            n += 1
    for m in acc: acc[m] /= n
    print(f"{tag:14s} {acc['zeros']:12.5f} {acc['reflect']:10.5f} {acc['replicate']:10.5f} "
          f"{acc['valid']:11.5f} {30*(acc['zeros']-acc['valid']):18.4f}")

print("\nB) sensitivity: corrupt the outer w px of the RENDER, watch SSIM vs PSNR")
tag, rd, gd = SETS[0]
D = None
torch.manual_seed(0)
for w, kind in [(1, "gray"), (3, "gray"), (5, "gray"), (1, "noise"), (5, "noise"),
                (5, "gtcopy"), (20, "gtcopy")]:
    S = P = 0.0; n = 0
    with torch.no_grad():
        for _, rp, gp in pairs(rd, gd):
            r = load(rp).to(DEV); g = load(gp).to(DEV)
            H, Wd = r.shape[-2:]
            if D is None or D.shape != (H, Wd): D = dm(H, Wd)
            m = (D < w).view(1, 1, H, Wd)
            if kind == "gray": rep = torch.full_like(r, 0.5)
            elif kind == "noise": rep = torch.rand_like(r)
            else: rep = g
            r = torch.where(m, rep, r)
            S += float(repo_ssim(r, g))
            P += 10 * np.log10(1 / max(((r - g) ** 2).mean().item(), 1e-12))
            n += 1
    print(f"  outer {w:2d}px -> {kind:7s}: SSIM {S/n:.5f}  PSNR {P/n:7.4f}   "
          f"(dSSIM {30*(S/n-0.86505):+.4f} pts, dPSNR {0.6*(P/n-24.9034):+.4f} pts)")

"""(a-3) Upper bound for ANY per-pixel photometric field, + does the lens-field warp
damage the frame boundary? + the remaining mild/wide border treatments."""
import sys, json, argparse
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from harness import *
import torch.nn.functional as Fn

ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="oracle_field"); a = ap.parse_args()


def dm(H, W):
    yy = torch.arange(H, device=DEV).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=DEV).view(1, W).expand(H, W)
    return torch.minimum(torch.minimum(yy, H - 1 - yy), torch.minimum(xx, W - 1 - xx))


def gk(sig):
    ks = int(2 * round(3 * sig) + 1)
    x = torch.arange(ks, dtype=torch.float32, device=DEV) - ks // 2
    k = torch.exp(-x ** 2 / (2 * sig ** 2)); k /= k.sum(); return k, ks


def blur(x, sig):
    if sig <= 0: return x
    k, ks = gk(sig); C = x.shape[1]
    x = Fn.conv2d(Fn.pad(x, (ks // 2, ks // 2, 0, 0), mode="replicate"),
                  k.view(1, 1, 1, ks).expand(C, 1, 1, ks), groups=C)
    return Fn.conv2d(Fn.pad(x, (0, 0, ks // 2, ks // 2), mode="replicate"),
                     k.view(1, 1, ks, 1).expand(C, 1, ks, 1), groups=C)


def sc(ps, ss, ls):
    P, S, L = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
    return P, S, L, score(P, S, L)


if a.stage == "oracle_field":
    # TEST-fitted per-pixel bias field: the ceiling for any spatial photometric correction.
    for s in SCENES[:1]:
        P = pairs(R_TEST.format(s=s), GT_TEST.format(s=s))
        imgs = [(load(rp).to(DEV), load(gp).to(DEV)) for _, rp, gp in P]
        n = len(imgs)
        idxA = list(range(0, n, 2)); idxB = list(range(1, n, 2))

        def field(idx):
            B = torch.zeros_like(imgs[0][0], dtype=torch.float64)
            for i in idx: B += (imgs[i][0] - imgs[i][1]).double()
            return (B / len(idx)).float()

        Ball = field(range(n)); BA = field(idxA); BB = field(idxB)
        print(f"\n=== {s} (test n={n}) per-pixel bias-field bound ===")
        for name, getB, sig in [("none", None, 0)] + \
                [(f"ORACLE(all,sig{g})", lambda i, g=g: blur(Ball, g), g) for g in (0, 4, 16)] + \
                [(f"CV 2-fold sig{g}", "cv", g) for g in (0, 4, 16)]:
            ps, ss, ls = [], [], []
            with torch.no_grad():
                for i, (r, g_) in enumerate(imgs):
                    if getB is None:
                        rr = r
                    elif getB == "cv":
                        Bf = blur(BB if i in idxA else BA, sig)
                        rr = r - Bf
                    else:
                        rr = r - getB(i)
                    rr = torch.round(rr.clamp(0, 1) * 255) / 255
                    ps.append(10 * np.log10(1 / max(((rr - g_) ** 2).mean().item(), 1e-12)))
                    ss.append(float(repo_ssim(rr, g_)))
                    ls.append(float(getvgg()(rr * 2 - 1, g_ * 2 - 1).item()))
            Pm, Sm, Lm, S0 = sc(ps, ss, ls)
            if name == "none": base = S0
            print(f"  {name:20s} PSNR {Pm:8.4f} SSIM {Sm:.5f} LPIPS {Lm:.5f} SCORE {S0:8.4f} "
                  f"dScore {S0-base:+.4f}", flush=True)
        del imgs; torch.cuda.empty_cache()

elif a.stage == "warp":
    # does applying the lens field damage the outer ring?
    for s in SCENES:
        for sub in ("test_poses_renders_png", "test_poses_renders_field"):
            rd = f"/mnt/d/avv/output/{s}_gsplatB9ut/{sub}"
            if not os.path.isdir(rd): continue
            acc = torch.zeros(21, dtype=torch.float64, device=DEV)
            acce = torch.zeros(21, dtype=torch.float64, device=DEV)
            cnt = torch.zeros(21, dtype=torch.float64, device=DEV)
            tS = tE = 0.0; nn = 0; D = None
            with torch.no_grad():
                for _, rp, gp in pairs(rd, GT_TEST.format(s=s)):
                    r = load(rp).to(DEV); g = load(gp).to(DEV)
                    sm = ssim_map(r, g).mean(1).reshape(-1).double()
                    er = ((r - g) ** 2).mean(1).reshape(-1).double()
                    H, W = r.shape[-2:]
                    if D is None: D = dm(H, W).clamp(max=20).reshape(-1)
                    acc.scatter_add_(0, D, sm); acce.scatter_add_(0, D, er)
                    cnt.scatter_add_(0, D, torch.ones_like(sm))
                    tS += float(sm.mean()); tE += float(er.mean()); nn += 1
            p = (acc / cnt).cpu().numpy(); pe = (acce / cnt).cpu().numpy()
            print(f"{s:9s} {sub[19:]:8s} SSIM_all {tS/nn:.5f} | ring SSIM d=0..4: "
                  + " ".join(f"{p[i]:.4f}" for i in range(5))
                  + f" | d>=20 {p[20]:.4f} | MSE d0 {pe[0]:.3e} d>=20 {pe[20]:.3e}", flush=True)

elif a.stage == "treat2":
    SETS = [("HCM0181/k4", K4, GT_TEST.format(s="HCM0181")),
            ("hcm0031/B9ut", R_TEST.format(s="hcm0031"), GT_TEST.format(s="hcm0031"))]
    treats = {"base": None,
              "blur0.7_w20": ("blur", 0.7, 20), "blur1.5_w20": ("blur", 1.5, 20),
              "shrink0.85_w10": ("shrink", 0.85, 10), "shrink0.85_w20": ("shrink", 0.85, 20),
              "shrink0.70_w20": ("shrink", 0.70, 20), "shrink0.85_w40": ("shrink", 0.85, 40)}
    D = None
    for tag, rd, gd in SETS:
        base = None
        for k, v in treats.items():
            ps, ss, ls = [], [], []
            with torch.no_grad():
                for _, rp, gp in pairs(rd, gd):
                    r = load(rp).to(DEV); g = load(gp).to(DEV)
                    H, W = r.shape[-2:]
                    if D is None: D = dm(H, W)
                    if v is not None:
                        kind, par, bw = v
                        w = blur((D < bw).float().view(1, 1, H, W), max(bw / 3.0, 1.0))
                        if kind == "blur":
                            r = r * (1 - w) + blur(r, par) * w
                        else:
                            lo = blur(r, 2.0); r = r * (1 - w) + (lo + par * (r - lo)) * w
                    r = torch.round(r.clamp(0, 1) * 255) / 255
                    ps.append(10 * np.log10(1 / max(((r - g) ** 2).mean().item(), 1e-12)))
                    ss.append(float(repo_ssim(r, g)))
                    ls.append(float(getvgg()(r * 2 - 1, g * 2 - 1).item()))
            Pm, Sm, Lm, S0 = sc(ps, ss, ls)
            if base is None: base = S0
            print(f"{tag:14s} {k:16s} PSNR {Pm:8.4f} SSIM {Sm:.5f} LPIPS {Lm:.5f} SCORE {S0:8.4f} "
                  f"dScore {S0-base:+.4f}", flush=True)

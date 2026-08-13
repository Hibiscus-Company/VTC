"""(a-followup) Where does the border deficit live -- systematic BIAS field or variance?

Fits a per-pixel photometric residual field on TRAIN views (clean holdout: applied to TEST),
plus fit-free border treatments (blur / contrast shrinkage in the border band).
"""
import sys, json, argparse
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from harness import *
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="diag")   # diag | fit | treat
ap.add_argument("--scenes", default=",".join(SCENES))
args = ap.parse_args()
SC = args.scenes.split(",")


def dist_map(H, W):
    yy = torch.arange(H, device=DEV).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=DEV).view(1, W).expand(H, W)
    return torch.minimum(torch.minimum(yy, H - 1 - yy), torch.minimum(xx, W - 1 - xx))


def gk(sig):
    ks = int(2 * round(3 * sig) + 1)
    x = torch.arange(ks, dtype=torch.float32, device=DEV) - ks // 2
    k = torch.exp(-x ** 2 / (2 * sig ** 2)); k /= k.sum()
    return k, ks


def blur(x, sig):
    if sig <= 0: return x
    k, ks = gk(sig); C = x.shape[1]
    kx = k.view(1, 1, 1, ks).expand(C, 1, 1, ks)
    x = Fn.conv2d(Fn.pad(x, (ks // 2, ks // 2, 0, 0), mode="replicate"), kx, groups=C)
    ky = k.view(1, 1, ks, 1).expand(C, 1, ks, 1)
    return Fn.conv2d(Fn.pad(x, (0, 0, ks // 2, ks // 2), mode="replicate"), ky, groups=C)


def accum_field(P):
    """mean residual (r-g) and mean squared residual, per pixel."""
    S = SQ = None; n = 0
    with torch.no_grad():
        for _, rp, gp in P:
            r = load(rp).to(DEV); g = load(gp).to(DEV); d = r - g
            if S is None:
                S = torch.zeros_like(d, dtype=torch.float64)
                SQ = torch.zeros_like(d, dtype=torch.float64)
            S += d.double(); SQ += (d * d).double(); n += 1
    return (S / n).float(), (SQ / n).float(), n


if args.stage == "diag":
    for s in SC:
        Ptr = pairs(R_TRAIN.format(s=s), GT_TRAIN.format(s=s))
        B, M, n = accum_field(Ptr)
        H, W = B.shape[-2:]
        D = dist_map(H, W)
        # bias estimate noise floor: var/n
        V = (M - B * B).clamp(min=0)
        noise = (V / n)
        print(f"\n=== {s} train n={n}  ({H}x{W}) ===")
        print(f"{'band':>10s} {'meanMSE':>10s} {'bias^2':>10s} {'bias^2_dbg':>11s} {'expl%':>7s}")
        bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1 << 20)]
        for lo, hi in bands:
            m = ((D >= lo) & (D < hi)).view(1, 1, H, W).expand_as(B)
            mse = float(M[m].mean())
            b2 = float((B[m] ** 2).mean())
            nz = float(noise[m].mean())
            b2d = max(b2 - nz, 0.0)   # debiased (subtract estimator noise)
            print(f"{f'{lo}-{hi if hi<1000 else 999}':>10s} {mse:10.3e} {b2:10.3e} {b2d:11.3e} "
                  f"{100*b2d/mse:7.3f}")
        # how much of the bias survives smoothing (i.e. is it a real smooth field?)
        for sig in (0, 2, 8, 32):
            Bs = blur(B, sig)
            print(f"  smooth sig={sig:3d}: ||B||^2 = {float((Bs**2).mean()):.4e}  "
                  f"(border d<20: {float((Bs**2)[:, :, :, :][ (D<20).view(1,1,H,W).expand_as(B)].mean()):.4e})")

elif args.stage == "fit":
    # fit residual field on TRAIN, apply to TEST -- clean holdout
    out = {}
    for s in SC:
        Ptr = pairs(R_TRAIN.format(s=s), GT_TRAIN.format(s=s))
        B, M, n = accum_field(Ptr)
        H, W = B.shape[-2:]
        D = dist_map(H, W)
        ted, egd = R_TEST.format(s=s), GT_TEST.format(s=s)

        variants = {"base": None}
        for sig in (2, 8, 32):
            variants[f"full_sig{sig}"] = ("sub", blur(B, sig), None)
        for sig in (2, 8):
            for bw in (10, 20, 40):
                w = (D < bw).float().view(1, 1, H, W)
                w = blur(w, 4)
                variants[f"band{bw}_sig{sig}"] = ("sub", blur(B, sig), w)

        for k, v in variants.items():
            ps, ss, ls = [], [], []
            with torch.no_grad():
                for _, rp, gp in pairs(ted, egd):
                    r = load(rp).to(DEV); g = load(gp).to(DEV)
                    if v is not None:
                        _, Bf, w = v
                        r = r - (Bf if w is None else Bf * w)
                    r = torch.round(r.clamp(0, 1) * 255) / 255
                    mse = ((r - g) ** 2).mean().item()
                    ps.append(10 * np.log10(1 / max(mse, 1e-12)))
                    ss.append(float(repo_ssim(r, g)))
                    ls.append(float(getvgg()(r * 2 - 1, g * 2 - 1).item()))
            P, S, L = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
            out[f"{s}/{k}"] = dict(PSNR=P, SSIM=S, LPIPS=L, SCORE=score(P, S, L))
            b = out[f"{s}/base"]
            print(f"{s:9s} {k:14s} PSNR {P:8.4f} SSIM {S:.5f} LPIPS {L:.5f} SCORE {score(P,S,L):8.4f} "
                  f"dScore {score(P,S,L)-b['SCORE']:+.4f}", flush=True)
    json.dump(out, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/field.json", "w"), indent=1)

elif args.stage == "treat":
    # fit-free border treatments on the k4 production ensemble + B9ut singles
    SETS = [("HCM0181/k4", K4, GT_TEST.format(s="HCM0181"))] + \
           [(f"{s}/B9ut", R_TEST.format(s=s), GT_TEST.format(s=s)) for s in SC]
    out = {}
    treats = {"base": None}
    for bw in (3, 5, 10, 20):
        for sig in (0.7, 1.5, 3.0):
            treats[f"blur{sig}_w{bw}"] = ("blur", sig, bw)
    for bw in (5, 10, 20):
        for al in (0.85, 0.7):
            treats[f"shrink{al}_w{bw}"] = ("shrink", al, bw)
    D = None
    for tag, rd, gd in SETS:
        for k, v in treats.items():
            ps, ss, ls = [], [], []
            with torch.no_grad():
                for _, rp, gp in pairs(rd, gd):
                    r = load(rp).to(DEV); g = load(gp).to(DEV)
                    H, W = r.shape[-2:]
                    if D is None: D = dist_map(H, W)
                    if v is not None:
                        kind, par, bw = v
                        w = (D < bw).float().view(1, 1, H, W)
                        w = blur(w, max(bw / 3.0, 1.0))
                        if kind == "blur":
                            r = r * (1 - w) + blur(r, par) * w
                        else:
                            lo = blur(r, 2.0)
                            r = r * (1 - w) + (lo + par * (r - lo)) * w
                    r = torch.round(r.clamp(0, 1) * 255) / 255
                    mse = ((r - g) ** 2).mean().item()
                    ps.append(10 * np.log10(1 / max(mse, 1e-12)))
                    ss.append(float(repo_ssim(r, g)))
                    ls.append(float(getvgg()(r * 2 - 1, g * 2 - 1).item()))
            P, S, L = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
            out[f"{tag}/{k}"] = dict(PSNR=P, SSIM=S, LPIPS=L, SCORE=score(P, S, L))
            b = out[f"{tag}/base"]
            print(f"{tag:14s} {k:16s} PSNR {P:8.4f} SSIM {S:.5f} LPIPS {L:.5f} "
                  f"SCORE {score(P,S,L):8.4f} dScore {score(P,S,L)-b['SCORE']:+.4f}", flush=True)
    json.dump(out, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/treat.json", "w"), indent=1)

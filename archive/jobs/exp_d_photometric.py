"""(d) NEGATIVE CONTROL: global brightness/gain/gamma fitted on TRAIN views, applied to TEST.

Also computes the TEST-fitted oracle so we can check the previously-reported +0.09 dB bound.
"""
import sys, json
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from harness import *

QUANT = True   # emulate the 8-bit write that production performs


def stack_stats(render_dir, gt_dir, limit=None):
    """Accumulate the sufficient statistics for per-channel least squares + gamma search."""
    P = pairs(render_dir, gt_dir)
    if limit: P = P[:limit]
    return P


def fit_affine(P, per_channel=True, use_bias=True, idx=None):
    """LS fit of gt ~ s*r + b, accumulated over the pair list (subset idx)."""
    sel = P if idx is None else [P[i] for i in idx]
    C = 3 if per_channel else 1
    Srr = torch.zeros(C, dtype=torch.float64, device=DEV)
    Srg = torch.zeros(C, dtype=torch.float64, device=DEV)
    Sr = torch.zeros(C, dtype=torch.float64, device=DEV)
    Sg = torch.zeros(C, dtype=torch.float64, device=DEV)
    N = 0.0
    with torch.no_grad():
        for _, rp, gp in sel:
            r = load(rp).to(DEV).double(); g = load(gp).to(DEV).double()
            if per_channel:
                rr = r.reshape(3, -1); gg = g.reshape(3, -1)
            else:
                rr = r.reshape(1, -1); gg = g.reshape(1, -1)
            Srr += (rr * rr).sum(1); Srg += (rr * gg).sum(1)
            Sr += rr.sum(1); Sg += gg.sum(1); N += rr.shape[1]
    if use_bias:
        det = Srr * N - Sr * Sr
        s = (Srg * N - Sr * Sg) / det
        b = (Srr * Sg - Sr * Srg) / det
    else:
        s = Srg / Srr
        b = torch.zeros_like(s)
    return s.float(), b.float()


def fit_gamma(P, idx=None, lo=0.85, hi=1.15, iters=24):
    """1-D golden-ish search on global gamma minimising MSE (with optimal gain re-fit)."""
    sel = P if idx is None else [P[i] for i in idx]
    imgs = []
    with torch.no_grad():
        for _, rp, gp in sel:
            imgs.append((load(rp).to(DEV), load(gp).to(DEV)))

    def mse_at(gam):
        tot = 0.0
        with torch.no_grad():
            for r, g in imgs:
                tot += float(((r.clamp(1e-6, 1) ** gam) - g).pow(2).mean())
        return tot / len(imgs)
    grid = np.linspace(lo, hi, 31)
    vals = [mse_at(float(x)) for x in grid]
    best = grid[int(np.argmin(vals))]
    # local refine
    for _ in range(3):
        step = (grid[1] - grid[0]) / 4
        cand = [best - 2 * step, best - step, best, best + step, best + 2 * step]
        v = [mse_at(float(x)) for x in cand]
        best = cand[int(np.argmin(v))]
    del imgs
    return float(best)


def apply_affine(r, s, b):
    out = r * s.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)
    out = out.clamp(0, 1)
    if QUANT:
        out = torch.round(out * 255) / 255
    return out


def apply_gamma(r, gam):
    out = r.clamp(1e-6, 1) ** gam
    out = out.clamp(0, 1)
    if QUANT:
        out = torch.round(out * 255) / 255
    return out


def evaluate(render_dir, gt_dir, tf):
    ps, ss, ls = [], [], []
    with torch.no_grad():
        for _, rp, gp in pairs(render_dir, gt_dir):
            r = load(rp).to(DEV); g = load(gp).to(DEV)
            r = tf(r)
            mse = ((r - g) ** 2).mean().item()
            ps.append(10 * np.log10(1 / max(mse, 1e-12)))
            ss.append(float(repo_ssim(r, g)))
            ls.append(float(getvgg()(r * 2 - 1, g * 2 - 1).item()))
    P, S, L = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
    return dict(PSNR=P, SSIM=S, LPIPS=L, SCORE=score(P, S, L))


res = {}
for s in SCENES:
    trd, tgd = R_TRAIN.format(s=s), GT_TRAIN.format(s=s)
    ted, egd = R_TEST.format(s=s), GT_TEST.format(s=s)
    Ptr = stack_stats(trd, tgd)
    print(f"\n=== {s}  (train pairs {len(Ptr)}) ===")

    fits = {}
    fits["gain1"] = ("affine",) + fit_affine(Ptr, per_channel=False, use_bias=False)
    fits["gain3"] = ("affine",) + fit_affine(Ptr, per_channel=True, use_bias=False)
    fits["affine3"] = ("affine",) + fit_affine(Ptr, per_channel=True, use_bias=True)
    gam = fit_gamma(Ptr)
    fits["gamma"] = ("gamma", gam, None)
    for k, v in fits.items():
        if v[0] == "affine":
            print(f"  fit {k:8s} s={v[1].cpu().numpy().round(5)} b={v[2].cpu().numpy().round(5)}")
        else:
            print(f"  fit {k:8s} gamma={v[1]:.4f}")

    base = evaluate(ted, egd, lambda r: (torch.round(r * 255) / 255) if QUANT else r)
    res[f"{s}/base"] = base
    print(f"  {'BASE (train-fit)':22s} PSNR {base['PSNR']:8.4f} SSIM {base['SSIM']:.5f} "
          f"LPIPS {base['LPIPS']:.5f} SCORE {base['SCORE']:8.4f}")
    for k, v in fits.items():
        if v[0] == "affine":
            tf = (lambda ss, bb: (lambda r: apply_affine(r, ss, bb)))(v[1], v[2])
        else:
            tf = (lambda gg: (lambda r: apply_gamma(r, gg)))(v[1])
        m = evaluate(ted, egd, tf)
        res[f"{s}/{k}"] = m
        print(f"  {k:22s} PSNR {m['PSNR']:8.4f} SSIM {m['SSIM']:.5f} LPIPS {m['LPIPS']:.5f} "
              f"SCORE {m['SCORE']:8.4f}  dScore {m['SCORE']-base['SCORE']:+.4f}")

    # TEST-FITTED ORACLE (upper bound, cheating)
    Pte = stack_stats(ted, egd)
    so, bo = fit_affine(Pte, per_channel=True, use_bias=True)
    m = evaluate(ted, egd, lambda r: apply_affine(r, so, bo))
    res[f"{s}/oracle_affine3"] = m
    print(f"  {'ORACLE affine3(test)':22s} PSNR {m['PSNR']:8.4f} SSIM {m['SSIM']:.5f} "
          f"LPIPS {m['LPIPS']:.5f} SCORE {m['SCORE']:8.4f}  dScore {m['SCORE']-base['SCORE']:+.4f}")

json.dump(res, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/photometric.json", "w"), indent=1)

# summary
print("\n=== SUMMARY dScore vs base ===")
keys = ["gain1", "gain3", "affine3", "gamma", "oracle_affine3"]
print(f"{'scene':10s} " + " ".join(f"{k:>16s}" for k in keys))
agg = {k: [] for k in keys}
for s in SCENES:
    b = res[f"{s}/base"]["SCORE"]
    row = []
    for k in keys:
        d = res[f"{s}/{k}"]["SCORE"] - b
        agg[k].append(d); row.append(f"{d:+16.4f}")
    print(f"{s:10s} " + " ".join(row))
print(f"{'MEAN':10s} " + " ".join(f"{np.mean(agg[k]):+16.4f}" for k in keys))

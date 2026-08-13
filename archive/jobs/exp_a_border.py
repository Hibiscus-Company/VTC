"""(a) SSIM frame-boundary structure: ring profile, oracle bound, legal treatments."""
import sys, json, argparse
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from harness import *

ap = argparse.ArgumentParser()
ap.add_argument("--render_dir", required=True)
ap.add_argument("--gt_dir", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--stage", default="profile")  # profile | oracle | treat
args = ap.parse_args()

PR = pairs(args.render_dir, args.gt_dir)
print(f"# {args.tag}  n={len(PR)}")

DMAX = 20


def dist_map(H, W, dev):
    yy = torch.arange(H, device=dev).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=dev).view(1, W).expand(H, W)
    return torch.minimum(torch.minimum(yy, H - 1 - yy), torch.minimum(xx, W - 1 - xx))


if args.stage == "profile":
    # ---- ring profile of the SSIM map + of squared error ----
    acc_s = torch.zeros(DMAX + 1, dtype=torch.float64, device=DEV)
    acc_e = torch.zeros(DMAX + 1, dtype=torch.float64, device=DEV)
    cnt = torch.zeros(DMAX + 1, dtype=torch.float64, device=DEV)
    tot_s = tot_e = 0.0
    int_s_num = torch.zeros((), dtype=torch.float64, device=DEV)
    int_s_den = 0.0
    D = None
    with torch.no_grad():
        for stem, rp, gp in PR:
            r = load(rp).to(DEV); g = load(gp).to(DEV)
            sm = ssim_map(r, g)           # 1,3,H,W
            er = (r - g) ** 2
            H, W = sm.shape[-2:]
            if D is None:
                D = dist_map(H, W, DEV)
                Dc = D.clamp(max=DMAX).reshape(-1)
            smm = sm.mean(1).reshape(-1).double()
            erm = er.mean(1).reshape(-1).double()
            acc_s.scatter_add_(0, Dc, smm)
            acc_e.scatter_add_(0, Dc, erm)
            cnt.scatter_add_(0, Dc, torch.ones_like(smm))
            tot_s += float(sm.mean()); tot_e += float(er.mean())
    n = len(PR)
    prof_s = (acc_s / cnt).cpu().numpy()
    prof_e = (acc_e / cnt).cpu().numpy()
    npx = (cnt / n).cpu().numpy()
    tot_s /= n; tot_e /= n
    print(f"# whole-image SSIM {tot_s:.6f}   MSE {tot_e:.6e}  ({H}x{W})")
    print(f"{'d':>3s} {'npix':>8s} {'frac%':>7s} {'SSIM(d)':>9s} {'MSE(d)':>10s} {'PSNR(d)':>8s}")
    for d in range(DMAX + 1):
        lab = f">={DMAX}" if d == DMAX else str(d)
        print(f"{lab:>3s} {npx[d]:8.0f} {100*npx[d]/(H*W):7.3f} {prof_s[d]:9.5f} {prof_e[d]:10.3e} "
              f"{10*np.log10(1/max(prof_e[d],1e-12)):8.3f}")
    # contribution of outer rings
    for w in (1, 2, 3, 5, 10):
        m = (D < w).reshape(-1)
        frac = float(m.double().mean())
        ring = float((acc_s.cpu().numpy()[:w] * 0).sum())  # placeholder
        ring_s = float(sum(acc_s.cpu().numpy()[:w]) / sum(cnt.cpu().numpy()[:w]))
        rest_s = (tot_s - frac * ring_s) / (1 - frac)
        print(f"# outer w={w:2d}: {100*frac:5.3f}% of pixels, SSIM_ring={ring_s:.5f}, "
              f"SSIM_interior={rest_s:.5f}, ring shifts total SSIM by "
              f"{frac*(ring_s-rest_s):+.6f} (= {30*frac*(ring_s-rest_s):+.4f} pts)")
    json.dump(dict(prof_s=prof_s.tolist(), prof_e=prof_e.tolist(), npx=npx.tolist(),
                   tot_s=tot_s, tot_e=tot_e, H=H, W=W),
              open(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/profile_{args.tag}.json", "w"))

elif args.stage == "oracle":
    # ---- ORACLE: paste GT into the outer ring of width w ----
    D = None
    res = {}
    widths = [0, 1, 2, 3, 5, 10, 20]
    for w in widths:
        ps, ss, ls = [], [], []
        with torch.no_grad():
            for stem, rp, gp in PR:
                r = load(rp).to(DEV); g = load(gp).to(DEV)
                if w > 0:
                    H, W = r.shape[-2:]
                    if D is None or D.shape != (H, W):
                        D = dist_map(H, W, DEV)
                    m = (D < w).view(1, 1, H, W)
                    r = torch.where(m, g, r)
                mse = ((r - g) ** 2).mean().item()
                ps.append(10 * np.log10(1 / max(mse, 1e-12)))
                ss.append(float(repo_ssim(r, g)))
                ls.append(float(getvgg()(r * 2 - 1, g * 2 - 1).item()))
        P, S, L = np.mean(ps), np.mean(ss), np.mean(ls)
        sc = score(P, S, L)
        res[w] = dict(PSNR=float(P), SSIM=float(S), LPIPS=float(L), SCORE=float(sc))
        b = res[0]
        print(f"oracle w={w:3d}  PSNR {P:8.4f} ({P-b['PSNR']:+7.4f})  SSIM {S:.5f} ({S-b['SSIM']:+.5f})  "
              f"LPIPS {L:.5f} ({L-b['LPIPS']:+.5f})  SCORE {sc:8.4f} ({sc-b['SCORE']:+.4f})")
    json.dump(res, open(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/oracle_{args.tag}.json", "w"), indent=1)

import os, json, glob, sys
import numpy as np

CACHE = "/mnt/d/avv/r42_bonsai78/a1_oracle/cache"
OUT = "/mnt/d/avv/r42_bonsai78/a1_oracle"


def sc(p, s, l):
    return 100.0 * (0.4 * (1 - l) + 0.3 * s + 0.3 * min(p / 50.0, 1.0))


def load():
    D = {}
    for f in glob.glob(os.path.join(CACHE, "*.json")):
        r = json.load(open(f))
        D.setdefault(r["stem"], {})[r["key"]] = r
    return D


def agg(D, keyfn, stems):
    P = np.mean([D[s][keyfn(s)]["psnr"] for s in stems])
    S = np.mean([D[s][keyfn(s)]["ssim"] for s in stems])
    L = np.mean([D[s][keyfn(s)]["lpips"] for s in stems])
    return P, S, L, sc(P, S, L)


def main():
    D = load()
    stems = sorted(D)
    print("frames:", len(stems))
    have = set.intersection(*[set(D[s]) for s in stems]) if stems else set()
    us_keys = sorted([k for k in have if k.startswith("us_")],
                     key=lambda k: (float(k.split("_")[1]), float(k.split("_")[2])))
    print("settings complete on all frames:", len(have))

    b = agg(D, lambda s: "base", stems)
    print(f"\n(1) BASELINE           PSNR {b[0]:8.4f}  SSIM {b[1]:.4f}  LPIPS {b[2]:.4f}  SCORE {b[3]:.4f}")
    assert max(D[s]["base"]["psnr"] for s in stems) < 50, "PSNR clamp active"

    print("\n(2) GLOBAL UNSHARP GRID  (Score / dScore vs baseline)")
    hdr = "sigma\\a"
    alphas = sorted({float(k.split("_")[2]) for k in us_keys})
    sigmas = sorted({float(k.split("_")[1]) for k in us_keys})
    print(f"{hdr:>8s} " + "".join(f"{a:>9.2f}" for a in [0.0] + alphas))
    rows = {}
    best = ("base", b)
    for sg in sigmas:
        line = f"{sg:8.1f} " + f"{b[3]:9.4f}"
        for a in alphas:
            k = f"us_{sg}_{a}"
            if k not in have:
                line += "        -"
                continue
            r = agg(D, lambda s, k=k: k, stems)
            rows[k] = r
            line += f"{r[3]:9.4f}"
            if r[3] > best[1][3]:
                best = (k, r)
        print(line)
    print(f"{'d':>8s} " + "".join(f"{0.0:>9.4f}" for a in [0.0]))
    for sg in sigmas:
        line = f"{sg:8.1f} " + f"{0.0:9.4f}"
        for a in alphas:
            k = f"us_{sg}_{a}"
            line += f"{rows[k][3]-b[3]:9.4f}" if k in rows else "        -"
        print(line)
    print(f"\nBEST GLOBAL: {best[0]}  PSNR {best[1][0]:.4f} SSIM {best[1][1]:.4f} LPIPS {best[1][2]:.4f} "
          f"SCORE {best[1][3]:.4f}  dScore {best[1][3]-b[3]:+.4f} "
          f"(dPSNR {best[1][0]-b[0]:+.4f} dSSIM {best[1][1]-b[1]:+.5f} dLPIPS {best[1][2]-b[2]:+.5f})")
    print("  term contributions: PSNR %+.4f  SSIM %+.4f  LPIPS %+.4f" % (
        0.6 * (best[1][0] - b[0]), 30 * (best[1][1] - b[1]), -40 * (best[1][2] - b[2])))

    # (3) oracle per frame over base + unsharp grid
    pick = {}
    for s in stems:
        cands = ["base"] + [k for k in us_keys]
        pick[s] = max(cands, key=lambda k: sc(D[s][k]["psnr"], D[s][k]["ssim"], D[s][k]["lpips"]))
    o = agg(D, lambda s: pick[s], stems)
    print(f"\n(3) ORACLE PER-FRAME UNSHARP  PSNR {o[0]:8.4f}  SSIM {o[1]:.4f}  LPIPS {o[2]:.4f}  SCORE {o[3]:.4f}  "
          f"dScore {o[3]-b[3]:+.4f}")
    print("  d: PSNR %+.4f dB  SSIM %+.5f  LPIPS %+.5f   contributions  P %+.4f  S %+.4f  L %+.4f" % (
        o[0] - b[0], o[1] - b[1], o[2] - b[2],
        0.6 * (o[0] - b[0]), 30 * (o[1] - b[1]), -40 * (o[2] - b[2])))

    for k in ("spec_pc", "spec_sh", "spec_pc_u8"):
        if k in have:
            r = agg(D, lambda s, k=k: k, stems)
            print(f"\n(4) {k:10s}  PSNR {r[0]:8.4f}  SSIM {r[1]:.4f}  LPIPS {r[2]:.4f}  SCORE {r[3]:.4f}  "
                  f"dScore {r[3]-b[3]:+.4f}")
            print("  d: PSNR %+.4f dB  SSIM %+.5f  LPIPS %+.5f   contributions  P %+.4f  S %+.4f  L %+.4f" % (
                r[0] - b[0], r[1] - b[1], r[2] - b[2],
                0.6 * (r[0] - b[0]), 30 * (r[1] - b[1]), -40 * (r[2] - b[2])))

    # u8 variants of best global
    for k in sorted(k for k in have if k.startswith("u8_")):
        r = agg(D, lambda s, k=k: k, stems)
        print(f"    {k:14s} SCORE {r[3]:.4f}  dScore {r[3]-b[3]:+.4f}")

    # combined oracle: per frame best of {base, unsharp grid, spec_pc, spec_sh}
    allk = ["base"] + us_keys + [k for k in ("spec_pc", "spec_sh") if k in have]
    pick2 = {s: max(allk, key=lambda k: sc(D[s][k]["psnr"], D[s][k]["ssim"], D[s][k]["lpips"]))
             for s in stems}
    o2 = agg(D, lambda s: pick2[s], stems)
    print(f"\n(3+4) ORACLE OVER ALL OPERATORS  SCORE {o2[3]:.4f}  dScore {o2[3]-b[3]:+.4f}")

    # CSV
    import csv
    path = os.path.join(OUT, "per_frame.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "base_score", "base_psnr", "base_ssim", "base_lpips",
                    "lapvar_gt", "lapvar_render", "lapvar_ratio_gt_over_render",
                    "oracle_sigma", "oracle_alpha", "oracle_score", "oracle_dscore",
                    "oracle_dpsnr", "oracle_dssim", "oracle_dlpips",
                    "specpc_score", "specpc_dscore", "specpc_dpsnr", "specpc_dssim", "specpc_dlpips",
                    "specsh_score", "specsh_dscore",
                    "bestglobal_key", "bestglobal_score", "bestglobal_dscore"])
        for s in stems:
            bb = D[s]["base"]
            bsc = sc(bb["psnr"], bb["ssim"], bb["lpips"])
            k = pick[s]
            sg, a = ("0", "0") if k == "base" else k.split("_")[1:]
            od = D[s][k]
            osc = sc(od["psnr"], od["ssim"], od["lpips"])
            row = [s, f"{bsc:.4f}", f"{bb['psnr']:.4f}", f"{bb['ssim']:.5f}", f"{bb['lpips']:.5f}",
                   f"{bb['lapvar_gt']:.6e}", f"{bb['lapvar_render']:.6e}",
                   f"{bb['lapvar_gt']/bb['lapvar_render']:.3f}",
                   sg, a, f"{osc:.4f}", f"{osc-bsc:.4f}",
                   f"{od['psnr']-bb['psnr']:.4f}", f"{od['ssim']-bb['ssim']:.5f}",
                   f"{od['lpips']-bb['lpips']:.5f}"]
            for kk in ("spec_pc", "spec_sh"):
                if kk in D[s]:
                    d = D[s][kk]
                    ssc = sc(d["psnr"], d["ssim"], d["lpips"])
                    if kk == "spec_pc":
                        row += [f"{ssc:.4f}", f"{ssc-bsc:.4f}", f"{d['psnr']-bb['psnr']:.4f}",
                                f"{d['ssim']-bb['ssim']:.5f}", f"{d['lpips']-bb['lpips']:.5f}"]
                    else:
                        row += [f"{ssc:.4f}", f"{ssc-bsc:.4f}"]
                else:
                    row += ["", "", "", "", ""] if kk == "spec_pc" else ["", ""]
            bd = D[s][best[0]]
            bsc2 = sc(bd["psnr"], bd["ssim"], bd["lpips"])
            row += [best[0], f"{bsc2:.4f}", f"{bsc2-bsc:.4f}"]
            w.writerow(row)
    print("\nCSV ->", path)

    # correlation diagnostics
    lg = np.array([D[s]["base"]["lapvar_gt"] for s in stems])
    lr = np.array([D[s]["base"]["lapvar_render"] for s in stems])
    oa = np.array([0.0 if pick[s] == "base" else float(pick[s].split("_")[2]) for s in stems])
    og = np.array([0.0 if pick[s] == "base" else float(pick[s].split("_")[1]) for s in stems])
    from scipy.stats import spearmanr, pearsonr
    print("\nlapvar_gt median %.4e   lapvar_render median %.4e  ratio median %.3f" % (
        np.median(lg), np.median(lr), np.median(lg / lr)))
    print("spearman(alpha_oracle, lapvar_gt) = %.3f ; (alpha, log ratio gt/ren) = %.3f ; (alpha, lapvar_render) = %.3f" % (
        spearmanr(oa, lg).correlation, spearmanr(oa, np.log(lg / lr)).correlation,
        spearmanr(oa, lr).correlation))
    print("spearman(sigma_oracle, lapvar_gt) = %.3f" % spearmanr(og, lg).correlation)
    dsc = np.array([sc(D[s][pick[s]]["psnr"], D[s][pick[s]]["ssim"], D[s][pick[s]]["lpips"])
                    - sc(D[s]["base"]["psnr"], D[s]["base"]["ssim"], D[s]["base"]["lpips"]) for s in stems])
    print("oracle per-frame dScore: mean %.4f  min %.4f  max %.4f  n>0.05: %d/%d" % (
        dsc.mean(), dsc.min(), dsc.max(), int((dsc > 0.05).sum()), len(dsc)))
    print("alpha histogram:", {a: int((oa == a).sum()) for a in sorted(set(oa.tolist()))})
    print("sigma histogram:", {g: int((og == g).sum()) for g in sorted(set(og.tolist()))})


if __name__ == "__main__":
    main()

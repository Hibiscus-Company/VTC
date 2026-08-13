"""A3 follow-up: WHY are the first 8 holes bad? test photometric offset, global
misregistration, and noise-corrected spectra. Also spatial split bad8 vs good20."""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_common import *  # noqa
import numpy as np
from scipy import stats
import cv2
import torch

OUT = "/mnt/d/avv/r42_bonsai78/a3_diag"
rows = json.load(open(f"{OUT}/bonsai_sr01_perframe.json"))
rows.sort(key=lambda r: r["frame"])


def immerkaer_sigma(g):
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(g, cv2.CV_32F, M)
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6.0)


def bandsof(g, rbin, NB, edges):
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))
    P = F.real ** 2 + F.imag ** 2
    s = np.bincount(rbin, weights=P.ravel(), minlength=NB)
    c = np.bincount(rbin, minlength=NB)
    rc = (np.arange(NB) + 0.5) / NB
    B = np.array([s[(rc >= a) & (rc < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    C = np.array([c[(rc >= a) & (rc < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    return B, C, float((win ** 2).sum())


edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NB = 256


def run(render_dir, gt_dir, tag, do_photo=True):
    pairs = pair_list(render_dir, gt_dir)
    pairs.sort(key=lambda t: t[0])
    rbin = None
    sB_R = np.zeros(5); sB_G = np.zeros(5); sB_Gn = np.zeros(5); Cb = None
    out = []
    for stem, rp, gp in pairs:
        R = load(rp); G = load(gp)
        gR = gray(R); gG = gray(G)
        if rbin is None:
            H, W = gG.shape
            fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None] / 0.5
            fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :] / 0.5
            rbin = np.clip((np.sqrt(fy**2 + fx**2) * NB).astype(np.int64), 0, NB - 1).ravel()
        bR, Cb, wss = bandsof(gR, rbin, NB, edges)
        bG, _, _ = bandsof(gG, rbin, NB, edges)
        sig = immerkaer_sigma(gG)
        # white-noise power per FFT bin = sigma^2 * sum(win^2); per band = that * nbins
        noise_band = sig ** 2 * wss * Cb
        bGn = np.maximum(bG - noise_band, 1e-9)
        sB_R += bR; sB_G += bG; sB_Gn += bGn
        rec = dict(stem=stem, sigma_gt=sig, sigma_render=immerkaer_sigma(gR))
        if do_photo:
            # optimal per-channel affine (gain,bias) render->GT, and resulting PSNR/SSIM
            r = R[0].reshape(3, -1).numpy(); g = G[0].reshape(3, -1).numpy()
            fix = np.empty_like(r)
            for c in range(3):
                A = np.stack([r[c], np.ones_like(r[c])], 1)
                sol, *_ = np.linalg.lstsq(A, g[c], rcond=None)
                fix[c] = np.clip(A @ sol, 0, 1)
            Rf = torch.from_numpy(fix.reshape(1, 3, *gG.shape))
            mse0 = float(((R - G) ** 2).mean()); mse1 = float(((Rf - G) ** 2).mean())
            rec["psnr_raw"] = 10 * np.log10(1 / mse0)
            rec["psnr_photofix"] = 10 * np.log10(1 / mse1)
            rec["ssim_raw"] = float(repo_ssim(R, G)); rec["ssim_photofix"] = float(repo_ssim(Rf, G))
            rec["dmean"] = float(gR.mean() - gG.mean())
            # global misregistration by phase correlation (sub-pixel)
            (dx, dy), resp = cv2.phaseCorrelate(gG.astype(np.float64), gR.astype(np.float64))
            rec["shift_x"] = dx; rec["shift_y"] = dy; rec["shift_mag"] = float(np.hypot(dx, dy))
            rec["pc_response"] = float(resp)
            # score after integer-shift alignment (upper bound on registration gain)
            best = (mse0, 0, 0)
            for sy in range(-3, 4):
                for sx in range(-3, 4):
                    Rs = torch.roll(R, shifts=(sy, sx), dims=(2, 3))
                    m = float(((Rs[:, :, 4:-4, 4:-4] - G[:, :, 4:-4, 4:-4]) ** 2).mean())
                    if m < best[0]:
                        best = (m, sy, sx)
            rec["psnr_shiftfix"] = 10 * np.log10(1 / best[0])
            rec["best_sy"], rec["best_sx"] = best[1], best[2]
            Rs = torch.roll(R, shifts=(best[1], best[2]), dims=(2, 3))
            rec["ssim_shiftfix"] = float(repo_ssim(Rs, G))
        out.append(rec)
        print(f"  {tag} {stem} sig_gt {sig:.5f} " +
              (f"dmean {rec['dmean']:+.4f} shift ({rec['shift_x']:+.2f},{rec['shift_y']:+.2f}) "
               f"bestint ({rec['best_sx']:+d},{rec['best_sy']:+d}) "
               f"P {rec['psnr_raw']:.2f}->photo {rec['psnr_photofix']:.2f} "
               f"->shift {rec['psnr_shiftfix']:.2f}" if do_photo else ""), flush=True)
    n = len(pairs)
    print(f"\n{tag} noise-corrected band ratios (render / noise-free GT):")
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        print(f"  {a:.1f}-{b:.1f} Nyq raw {sB_R[i]/sB_G[i]:.4f}  "
              f"noise-corrected {sB_R[i]/sB_Gn[i]:.4f}  "
              f"(noise share of GT band = {100*(1-sB_Gn[i]/sB_G[i]):.1f}%)")
    np.save(f"{OUT}/{tag}_bands_noisecorr.npy",
            np.stack([sB_R / n, sB_G / n, sB_Gn / n]))
    json.dump(out, open(f"{OUT}/{tag}_cause.json", "w"), indent=1)
    if out and "psnr_raw" in out[0]:
        with open(f"{OUT}/{tag}_cause.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader()
            [w.writerow(r) for r in out]
    return out


print("### BONSAI ###")
ob = run("/mnt/d/avv/r36_shape/sr01/eval_png", "/mnt/d/avv/evalsplit/bonsai/eval_gt", "bonsai_sr01")
print("\n### HCM0421 (reference tower) ###")
oh = run("/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/evalsplit/HCM0421/eval_gt",
         "HCM0421_evalgen", do_photo=False)

sig_b = np.array([r["sigma_gt"] for r in ob]); sig_h = np.array([r["sigma_gt"] for r in oh])
print(f"\nGT sensor-noise sigma (Immerkaer, [0,1] gray): bonsai median {np.median(sig_b):.5f} "
      f"({np.median(sig_b)*255:.2f}/255) | HCM0421 median {np.median(sig_h):.5f} "
      f"({np.median(sig_h)*255:.2f}/255)")

# summarise the causal probes
ob.sort(key=lambda r: r["stem"])
idx8 = np.arange(8); idx20 = np.arange(8, 28)
for nm in ["psnr_raw", "psnr_photofix", "psnr_shiftfix", "ssim_raw", "ssim_photofix",
           "ssim_shiftfix", "dmean", "shift_mag"]:
    v = np.array([r[nm] for r in ob])
    print(f"  {nm:14s} first8 {v[idx8].mean():+8.4f}  last20 {v[idx20].mean():+8.4f}  "
          f"all {v.mean():+8.4f}")
dP_photo = np.array([r["psnr_photofix"] - r["psnr_raw"] for r in ob])
dS_photo = np.array([r["ssim_photofix"] - r["ssim_raw"] for r in ob])
dP_shift = np.array([r["psnr_shiftfix"] - r["psnr_raw"] for r in ob])
dS_shift = np.array([r["ssim_shiftfix"] - r["ssim_raw"] for r in ob])
print(f"\n  per-image affine photometric fix: dPSNR {dP_photo.mean():+.4f} dSSIM {dS_photo.mean():+.4f} "
      f"-> dScore {(0.6*dP_photo+30*dS_photo).mean():+.4f} (first8 "
      f"{(0.6*dP_photo+30*dS_photo)[idx8].mean():+.4f})")
print(f"  best integer shift fix        : dPSNR {dP_shift.mean():+.4f} dSSIM {dS_shift.mean():+.4f} "
      f"-> dScore {(0.6*dP_shift+30*dS_shift).mean():+.4f} (first8 "
      f"{(0.6*dP_shift+30*dS_shift)[idx8].mean():+.4f})")

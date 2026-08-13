"""P3: SPECTRUM MATCHING.  Our renders are 1-7 dB short of the GT's power in the mid band and
1-3 dB OVER it in the top octave.  Fit a zero-phase radial gain g(f) on one half of the images,
apply it to the other half, and score all three metrics separately.

Family:  G_lambda(f) = g_match(f) ** lambda ,   g_match = sqrt(S_GG / S_RR)
  lambda = 0    -> identity (current shipping state)
  lambda = 1    -> exact power-spectrum match to the GT
Also:  g_wiener  = Re(S_RG)/S_RR  (the MSE-optimal zero-phase radial filter; the graveyard's
       'MSE-optimal kernel' lives here) and a 'trim-only' variant that applies g_wiener ONLY
       above f0 (kill the aliasing excess) and leaves the rest alone.

2-fold: gains are fitted on fold A and scored on fold B and vice versa, so nothing is self-fitted.
"""
import os, sys, json
import numpy as np
import torch
import mlib

torch.set_num_threads(int(os.environ.get("NT", "8")))
NB = 64
BINS = np.linspace(0, 0.7072, NB + 1)


def rad_idx(H, W):
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    return np.clip(np.digitize(r, BINS) - 1, 0, NB - 1)


def stats(imgs_r, imgs_g):
    """accumulate radial S_RR, S_GG, Re S_RG over a list of (HxWx3 uint8) pairs"""
    SRR = np.zeros(NB); SGG = np.zeros(NB); SRG = np.zeros(NB); CNT = np.zeros(NB)
    for r8, g8 in zip(imgs_r, imgs_g):
        for c in range(3):
            R = np.fft.fft2(r8[..., c].astype(np.float64) / 255.0)
            G = np.fft.fft2(g8[..., c].astype(np.float64) / 255.0)
            idx = rad_idx(*R.shape).ravel()
            np.add.at(SRR, idx, (R * np.conj(R)).real.ravel())
            np.add.at(SGG, idx, (G * np.conj(G)).real.ravel())
            np.add.at(SRG, idx, (R * np.conj(G)).real.ravel())
            np.add.at(CNT, idx, 1)
    CNT = np.maximum(CNT, 1)
    return SRR / CNT, SGG / CNT, SRG / CNT


def apply_radial(img8, gain):
    """zero-phase radial filter, reflect-padded to kill FFT wrap-around."""
    P = 48
    x = np.pad(img8.astype(np.float64) / 255.0, ((P, P), (P, P), (0, 0)), mode="reflect")
    H, W, _ = x.shape
    idx = rad_idx(H, W)
    Gmap = gain[idx]
    out = np.empty_like(x)
    for c in range(3):
        out[..., c] = np.fft.ifft2(np.fft.fft2(x[..., c]) * Gmap).real
    return np.clip(out[P:-P, P:-P] * 255.0, 0, 255).astype(np.uint8)


def main():
    key = sys.argv[1]
    NIMG = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    CASES = {
        "chair": ("/mnt/d/avv/chair_eval/base60k/eval_png", "/mnt/d/avv/evalsplit/chair/eval_gt", []),
        "HCM0421": ("/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/evalsplit/HCM0421/eval_gt",
                    ["DJI_20241230093301_0003_V"]),
        "bonsai": ("/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_png", "/mnt/d/avv/evalsplit/bonsai2/eval_gt", []),
    }
    rd, gd, skip = CASES[key]
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
    files = [f for f in sorted(os.listdir(rd)) if os.path.splitext(f)[0] not in skip]
    files = files[:: max(1, len(files) // NIMG)][:NIMG]
    R8 = [mlib.load_u8(os.path.join(rd, f)) for f in files]
    G8 = [mlib.load_u8(os.path.join(gd, gt_by[os.path.splitext(f)[0]])) for f in files]

    lp = mlib.LP("cpu")
    folds = [(list(range(0, len(files), 2)), list(range(1, len(files), 2)))]
    folds.append((folds[0][1], folds[0][0]))

    LAM = [0.0, 0.25, 0.5, 0.75, 1.0]
    names = ["identity"] + [f"match^{l}" for l in LAM[1:]] + ["wiener", "trim_only_f0.40",
                                                              "trim_only_f0.45", "match0.5_trim"]
    acc = {n: [0.0, 0.0, 0.0, 0] for n in names}
    for fit, test in folds:
        SRR, SGG, SRG = stats([R8[i] for i in fit], [G8[i] for i in fit])
        gmatch = np.sqrt(np.maximum(SGG, 1e-30) / np.maximum(SRR, 1e-30))
        gwien = SRG / np.maximum(SRR, 1e-30)
        gmatch = np.clip(gmatch, 0.3, 3.0)
        gwien = np.clip(gwien, 0.0, 3.0)
        fc = 0.5 * (BINS[:-1] + BINS[1:])
        gains = {"identity": np.ones(NB)}
        for l in LAM[1:]:
            gains[f"match^{l}"] = gmatch ** l
        gains["wiener"] = gwien
        for f0 in (0.40, 0.45):
            g = np.ones(NB); m = fc > f0; g[m] = gwien[m]
            gains[f"trim_only_f{f0:.2f}"] = g
        g = gmatch ** 0.5
        m = fc > 0.42
        g[m] = np.minimum(g[m], gwien[m])
        gains["match0.5_trim"] = g
        if fit is folds[0][0]:
            print("radial gain table (fold A):  f  gmatch  gwiener")
            for i in range(0, NB, 4):
                print(f"   {fc[i]:.3f}  {gmatch[i]:6.3f}  {gwien[i]:6.3f}")
        for i in test:
            g = mlib.to_t(G8[i])
            for n in names:
                y = mlib.to_t(apply_radial(R8[i], gains[n]))
                acc[n][0] += mlib.psnr(y, g)
                acc[n][1] += float(mlib.ssim(y, g))
                acc[n][2] += lp(y, g)
                acc[n][3] += 1
            print(f"   img {i} done", flush=True)

    print(f"\n=== P3 {key}  n={len(files)} (2-fold, gains never self-fitted) ===")
    rows = [(n, *[v / acc[n][3] for v in acc[n][:3]]) for n in names]
    rows = [(n, P, S, L, mlib.score(P, S, L)) for n, P, S, L in rows]
    ref = rows[0]
    for n, P, S, L, sc in rows:
        print(f"{n:18s} PSNR {P:7.4f}({P-ref[1]:+.4f})  SSIM {S:.5f}({S-ref[2]:+.5f})  "
              f"LPIPS {L:.5f}({L-ref[3]:+.5f})  SCORE {sc:8.4f}  d {sc-ref[4]:+.4f}")
    json.dump(rows, open(f"/mnt/d/avv/metric_probe/p3_{key}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

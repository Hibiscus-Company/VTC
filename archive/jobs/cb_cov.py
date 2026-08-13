#!/usr/bin/env python
"""COMBINER LENS -- the ORACLE bound on linear member weighting.

The pixel mean is the equal-weight linear combiner.  The MSE-optimal linear combiner with
weights summing to 1 is GLS:  w* = C^-1 1 / (1' C^-1 1),  C = member ERROR covariance
against the real test GT.  Fitting C on the very views we score is an ORACLE -- it upper
bounds what ANY weight-tuning scheme (family weights included) can buy on the PSNR term.
If the oracle gain is small, the whole weighting axis is dead and no honest scheme can win.

Also reports a 2-fold split-half HONEST weight transfer, and the family-block structure of
C (which is what makes hierarchical / family weighting either right or pointless).
"""
import os, sys, json
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from cb_run import POOLS, mdir, GTD
Image.MAX_IMAGE_PIXELS = None
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"


def main():
    pool = sys.argv[1] if len(sys.argv) > 1 else "A"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    big, small = POOLS[pool]
    VAR = big + small; k = len(VAR); NB = len(big)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(mdir(v), s + ".png")) for v in VAR))[:n]
    Cs = []
    for s in stems:
        X = torch.stack([torch.from_numpy(
            np.asarray(Image.open(os.path.join(mdir(v), s + ".png")).convert("RGB"),
                       np.float32) / 255.).permute(2, 0, 1) for v in VAR], 0).to(DEV)
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"),
                                        np.float32) / 255.).permute(2, 0, 1).to(DEV)
        R = (X - g.unsqueeze(0)).reshape(k, -1)
        Cs.append((R @ R.T / R.shape[1]).double().cpu().numpy())
        del X, g, R
    Cs = np.array(Cs)                                    # (n,k,k) per-view error covariance
    C = Cs.mean(0)
    one = np.ones(k)

    def gls(Cm):
        v = np.linalg.solve(Cm, one)
        return v / v.sum()

    def mse(w, Cm):
        return float(w @ Cm @ w)

    w_flat = one / k
    w_star = gls(C)
    # family-weight family: total weight ws on the small family
    grid = np.linspace(0.0, 0.7, 71)
    fam = []
    for ws in grid:
        w = np.concatenate([np.full(NB, (1 - ws) / NB), np.full(k - NB, ws / (k - NB))])
        fam.append(mse(w, C))
    fam = np.array(fam)
    ib = int(fam.argmin())
    m_flat, m_star = mse(w_flat, C), mse(w_star, C)
    print(f"pool={pool}  k={k} (big {NB} / small {k-NB})  views={len(stems)}")
    print(f"members: {VAR}")
    print(f"\nflat-mean MSE      {m_flat:.6e}   PSNR {10*np.log10(1/m_flat):7.4f}")
    print(f"ORACLE GLS   MSE   {m_star:.6e}   PSNR {10*np.log10(1/m_star):7.4f}"
          f"   dPSNR {10*np.log10(m_flat/m_star):+.4f} dB  -> dScore {0.6*10*np.log10(m_flat/m_star):+.4f}")
    print(f"ORACLE GLS weights: " + " ".join(f"{v:+.4f}" for v in w_star))
    print(f"  -> small-family total weight {w_star[NB:].sum():+.4f}  (flat = {(k-NB)/k:.4f})")
    print(f"\nbest FAMILY weight ws={grid[ib]:.3f}  MSE {fam[ib]:.6e}"
          f"  dPSNR {10*np.log10(m_flat/fam[ib]):+.4f} dB"
          f"  -> dScore {0.6*10*np.log10(m_flat/fam[ib]):+.4f}")
    print("  ws:  " + "  ".join(f"{grid[i]:.2f}" for i in range(0, 71, 5)))
    print("  dPS: " + "  ".join(f"{10*np.log10(m_flat/fam[i]):+.3f}" for i in range(0, 71, 5)))
    # honest split-half transfer of the GLS weights
    h = len(stems) // 2
    d = []
    for a, b in ((slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))):
        wf = gls(Cs[a].mean(0)); Ct = Cs[b].mean(0)
        d.append(10 * np.log10(mse(w_flat, Ct) / mse(wf, Ct)))
    print(f"\nHONEST split-half GLS transfer: dPSNR {np.mean(d):+.4f} dB "
          f"(folds {d[0]:+.4f}/{d[1]:+.4f}) -> dScore {0.6*np.mean(d):+.4f}")
    # block structure
    R = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    bb = R[:NB, :NB][np.triu_indices(NB, 1)].mean()
    ss = R[NB:, NB:][np.triu_indices(k - NB, 1)].mean()
    bs = R[:NB, NB:].mean()
    print(f"\nerror-correlation blocks: big-big {bb:.4f}  small-small {ss:.4f}  cross {bs:.4f}")
    print(f"per-member error PSNR: " +
          " ".join(f"{v}:{10*np.log10(1/C[i,i]):.2f}" for i, v in enumerate(VAR)))
    json.dump(dict(pool=pool, vars=VAR, nbig=NB, C=C.tolist(), w_star=w_star.tolist(),
                   fam_grid=grid.tolist(), fam_mse=fam.tolist(), honest=d,
                   m_flat=m_flat, m_star=m_star),
              open(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/cb_cov_{pool}.json", "w"))


if __name__ == "__main__":
    main()

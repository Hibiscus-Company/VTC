#!/usr/bin/env python
"""D5: is the residual MSE misregistration, photometric, or irreducible?

D4 showed error is concentrated in the top 1% of pixels (35.6% of SE) and the
image border carries 1.8x the center's error density. Three candidate causes,
three ORACLE upper bounds (fit against PUBLIC test GT -- diagnosis only; any
real fix must be trainable from train data alone, like D1's oracle color test):

  (a) GEOMETRIC: a small global shift, or a dense local flow, snaps render->GT.
      -> cause is pose / lens-distortion error. Fixable (pose refine, k2 term).
  (b) PHOTOMETRIC-RADIAL: a radial gain/offset curve (vignetting) explains it.
      A GLOBAL affine (D1, +0.09 dB) cannot see this -- it is spatially varying.
  (c) IRREDUCIBLE: neither helps -> content that simply differs between shots
      (moving vegetation, people, specular). Nothing to win; PSNR is capped.

Each oracle is an UPPER BOUND on what the corresponding real fix could buy.
"""
import argparse, os, sys
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None


def psnr_db(mse):
    return 10.0 * np.log10(1.0 / max(float(mse), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt)}
    rends = sorted(f for f in os.listdir(args.render)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[: args.limit]

    base, o_shift, o_flow, o_rad, o_flowrad = [], [], [], [], []
    shifts = []
    # radial profiles, 10 bins
    NB = 10
    se_r = np.zeros(NB); n_r = np.zeros(NB)
    rl_r = np.zeros(NB); gl_r = np.zeros(NB)

    for f in rends:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by_stem:
            continue
        r = np.asarray(Image.open(os.path.join(args.render, f)).convert("RGB"),
                       dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(args.gt, gt_by_stem[stem])).convert("RGB"),
                       dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        base.append(((r - g) ** 2).mean())

        # ---------- radial bookkeeping ----------
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        rad = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        rad /= rad.max()
        b = np.clip((rad * NB).astype(int), 0, NB - 1)
        se_pix = ((r - g) ** 2).mean(axis=2)
        for i in range(NB):
            m = b == i
            se_r[i] += se_pix[m].sum(); n_r[i] += m.sum()
            rl_r[i] += r.mean(axis=2)[m].sum(); gl_r[i] += g.mean(axis=2)[m].sum()

        # ---------- (a1) oracle GLOBAL integer shift, +-4 px ----------
        rg = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
        gg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
        (dx, dy), _ = cv2.phaseCorrelate(rg.astype(np.float64), gg.astype(np.float64))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        r_sh = cv2.warpAffine(r, M, (W, H), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)
        o_shift.append(((r_sh - g) ** 2).mean())
        shifts.append((dx, dy))

        # ---------- (a2) oracle DENSE LOCAL FLOW (upper bound on registration) ----------
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc((gg * 255).astype(np.uint8), (rg * 255).astype(np.uint8), None)
        # cap to small motion: this is registration, not synthesis
        flow = np.clip(flow, -6, 6)
        mapx = (xx + flow[..., 0]).astype(np.float32)
        mapy = (yy + flow[..., 1]).astype(np.float32)
        r_fl = cv2.remap(r, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        o_flow.append(((r_fl - g) ** 2).mean())

        # ---------- (b) oracle RADIAL gain+offset (vignetting), 16 rings ----------
        NR = 16
        br = np.clip((rad * NR).astype(int), 0, NR - 1)
        r_rad = r.copy()
        for i in range(NR):
            m = br == i
            if m.sum() < 16:
                continue
            x = r[m].reshape(-1); y = g[m].reshape(-1)
            A = np.stack([x, np.ones_like(x)], 1)
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            r_rad[m] = (r[m] * sol[0] + sol[1])
        r_rad = np.clip(r_rad, 0, 1)
        o_rad.append(((r_rad - g) ** 2).mean())

        # ---------- (a2)+(b) combined: flow then radial ----------
        r_fr = r_fl.copy()
        for i in range(NR):
            m = br == i
            if m.sum() < 16:
                continue
            x = r_fl[m].reshape(-1); y = g[m].reshape(-1)
            A = np.stack([x, np.ones_like(x)], 1)
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            r_fr[m] = (r_fl[m] * sol[0] + sol[1])
        o_flowrad.append(((np.clip(r_fr, 0, 1) - g) ** 2).mean())

    if not base:
        sys.exit("no images matched")

    def mp(lst):  # mean of per-image PSNR (what the scorer does)
        return float(np.mean([psnr_db(m) for m in lst]))

    b0 = mp(base)
    print(f"\n===== D5 {args.tag}  ({len(base)} images) =====")
    print(f"baseline                       {b0:7.4f} dB")
    for lab, lst in (("oracle GLOBAL shift (a1)", o_shift),
                     ("oracle DENSE FLOW  (a2)", o_flow),
                     ("oracle RADIAL gain (b) ", o_rad),
                     ("oracle FLOW + RADIAL   ", o_flowrad)):
        v = mp(lst)
        print(f"{lab}       {v:7.4f} dB   ({v - b0:+.4f} dB = {0.6 * (v - b0):+.4f} score pts)")

    sx = np.array([s[0] for s in shifts]); sy = np.array([s[1] for s in shifts])
    print(f"\nglobal shift found: dx {sx.mean():+.3f}+-{sx.std():.3f}  "
          f"dy {sy.mean():+.3f}+-{sy.std():.3f} px  (|dx|max {np.abs(sx).max():.2f})")

    print("\n-- radial profile (0=center, 9=corner) --")
    print("  bin   SE/px    rel    render_lum  gt_lum   bias")
    gm = se_r.sum() / n_r.sum()
    for i in range(NB):
        if n_r[i] == 0:
            continue
        d = se_r[i] / n_r[i]
        rl = rl_r[i] / n_r[i]; gl = gl_r[i] / n_r[i]
        print(f"   {i}   {d:.6f}  {d / gm:5.2f}x    {rl:.4f}    {gl:.4f}  {rl - gl:+.4f}")


if __name__ == "__main__":
    main()

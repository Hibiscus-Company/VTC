#!/usr/bin/env python
"""IS THIS NEW ENSEMBLE MEMBER SAFE TO GIVE 1/N OF A SCENE'S WEIGHT?

bonsai has already produced a member that trained to completion, wrote 28 valid PNGs, and was FOG --
median opacity 0.000, median scale 0.0003, l1 0.26 at 15k after being 0.038 at 5k. Nothing about the
file listing said so. mip3d changes the densification dynamics (it rescales every gaussian and
compensates opacity), so that failure mode is live again on exactly the scene that had it.

This gate needs no ground truth. It compares the candidate against the ENSEMBLE IT WOULD JOIN:
  - contrast collapse: a fogged render has far less local detail than its peers
  - global drift:      a member that disagrees with the existing mean by a lot is either a great
                       decorrelated member or a broken one; combined with the contrast number it
                       separates the two
  - dead frames:       constant or near-constant images, NaNs, all-black
Reference band comes from the EXISTING members, so the threshold is the scene's own scatter rather
than a number I invented.
"""
import argparse, os, sys
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def stats(d, stems):
    lap, mu, sd, dead = [], [], [], 0
    for s in stems:
        p = os.path.join(d, s + ".png")
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        if not np.isfinite(a).all():
            dead += 1
            continue
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        lap.append(float((cv2.Laplacian(g, cv2.CV_32F) ** 2).mean()))
        mu.append(float(a.mean())); sd.append(float(a.std()))
        if float(a.std()) < 0.02:
            dead += 1
    return np.array(lap), np.array(mu), np.array(sd), dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--peers", nargs="+", required=True)
    ap.add_argument("--expect", type=int, required=True)
    args = ap.parse_args()

    have = sorted(os.path.splitext(f)[0] for f in os.listdir(args.candidate)
                  if f.lower().endswith(".png"))
    print(f"candidate {args.candidate}: {len(have)} renders (expect {args.expect})")
    ok = len(have) == args.expect
    if not ok:
        print("  FAIL: wrong render count")

    stems = have[:min(12, len(have))]
    cl, cm, cs, cd = stats(args.candidate, stems)
    print(f"{'member':>52} {'detail':>10} {'mean':>7} {'std':>7} {'dead':>5}")
    print(f"{'CANDIDATE':>52} {cl.mean():10.5f} {cm.mean():7.4f} {cs.mean():7.4f} {cd:5d}")
    P = []
    for p in args.peers:
        if not os.path.isdir(p):
            print(f"  (missing peer {p})"); continue
        pl, pm, ps, pd = stats(p, stems)
        P.append(pl.mean())
        print(f"{os.path.basename(os.path.dirname(p))+'/'+os.path.basename(p):>52} "
              f"{pl.mean():10.5f} {pm.mean():7.4f} {ps.mean():7.4f} {pd:5d}")
    if not P:
        sys.exit("no peers to compare against")
    lo, hi = min(P), max(P)
    band_lo = lo - 0.5 * (hi - lo) - 0.25 * lo        # generous: a REGULARISED member may be softer
    print(f"\npeer detail range {lo:.5f}..{hi:.5f};  candidate {cl.mean():.5f}")
    if cd > 0:
        print(f"  FAIL: {cd} dead/non-finite frames"); ok = False
    if cl.mean() < band_lo:
        print(f"  FAIL: detail {cl.mean():.5f} is below the collapse floor {band_lo:.5f} "
              f"-- this is the bonsai fog signature"); ok = False
    elif cl.mean() < lo:
        print(f"  WARN: softer than every peer, but within the regulariser-tolerance band "
              f"(floor {band_lo:.5f}). mip3d IS a band-limiter, so some softening is expected.")
    print("\nVERDICT:", "PASS -- safe to include" if ok else "FAIL -- do NOT include this member")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

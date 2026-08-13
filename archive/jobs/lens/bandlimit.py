#!/usr/bin/env python
"""CAN WE WARP WITHOUT PAYING THE RESAMPLING TAX AT ALL?

cubic -> lanczos4 bought +0.108/scene by cutting the warp's high-frequency loss from 5.2%
to 2.4%. The remaining 2.4% is worth roughly another +0.09/scene at the same exchange rate,
IF it can be reached. It can only be reached by a reconstruction kernel closer to the ideal
sinc than lanczos4's 8-tap window.

THE ROUTE: zero-padding the spectrum is an EXACT band-limited upsample -- it adds no
frequencies above the original Nyquist and destroys none below it. Upsample the render by k
that way, then sample the fine grid at k*(x+dx). On the fine grid the signal is oversampled
k-fold, so an ordinary cubic kernel is only being asked about frequencies up to 1/k of ITS
Nyquist, where its transfer function is essentially flat. The warp becomes near-ideal for the
price of one FFT.

Two things could kill it and both are checked here rather than assumed:
  - Gibbs ringing at the frame border (the DFT continues the image periodically). Reflect-pad
    before the transform, crop after; pad to an ODD size so fftshift is symmetric and there is
    no Nyquist-term ambiguity to fudge.
  - Cost. k=4 on a 1320x989 frame is a 4468x5792 transform per channel.

GATE (this script): round-trip isolation, no GT involved -- warp by +f, warp back by -f, and
measure what came back. It cannot tell you a warp is CORRECT, only how much detail it
destroys, which is exactly the quantity in question. Only if it beats lanczos4 here does the
full production harness run get spent on it.
"""
import argparse, os, sys, time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fieldlib import LooPool, upsample, warp as base_warp

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))


def _odd_pad(n, pad):
    """pad so that n + 2*pad is ODD -> fftshift is exactly symmetric, no Nyquist term"""
    lo = pad
    hi = pad + (0 if (n + 2 * pad) % 2 == 1 else 1)
    return lo, hi


def fft_upsample(chan, k, pad=48):
    """exact band-limited k-fold upsample of a single float32 channel"""
    from scipy import fft as sfft
    H, W = chan.shape
    (t, b), (l, r) = _odd_pad(H, pad), _odd_pad(W, pad)
    p = np.pad(chan, ((t, b), (l, r)), mode="reflect").astype(np.float32)
    Hp, Wp = p.shape
    F = sfft.fftshift(sfft.fft2(p))
    Hn, Wn = Hp * k, Wp * k
    G = np.zeros((Hn, Wn), dtype=F.dtype)
    # align DC, do NOT centre the block: after fftshift the DC bin sits at index N//2 for
    # both parities, and Hp odd with k even makes (Hn-Hp)//2 land one bin low. One bin of
    # offset is a linear phase ramp across the spectrum, i.e. a half-pixel shift plus a
    # real/imaginary scramble -- it does not look like a subtle bug, it destroys the image.
    y0, x0 = Hn // 2 - Hp // 2, Wn // 2 - Wp // 2
    G[y0:y0 + Hp, x0:x0 + Wp] = F
    out = sfft.ifft2(sfft.ifftshift(G)).real.astype(np.float32) * (k * k)
    # the original sample (i,j) sits at fine index (k*(i+t), k*(j+l))
    return out, t * k, l * k


def warp_bandlimited(img, fu, k=2, fine="cubic", pad=48):
    """warp img by the full-res displacement fu, resampling on an exact k-fold fine grid"""
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fl = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR,
          "lanczos": cv2.INTER_LANCZOS4}[fine]
    out = np.empty_like(img)
    for c in range(3):
        fineimg, oy, ox = fft_upsample(img[..., c], k, pad)
        mx = ((xx + fu[..., 0]) * k + ox).astype(np.float32)
        my = ((yy + fu[..., 1]) * k + oy).astype(np.float32)
        out[..., c] = cv2.remap(fineimg, mx, my, fl, borderMode=cv2.BORDER_REFLECT)
        del fineimg
    return out


def hf_energy(img):
    g = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2GRAY)
    return float((cv2.Laplacian(g, cv2.CV_32F) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", default="/mnt/d/avv/prodharness/k4/png")
    ap.add_argument("--tag", default="HCM0181")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    cache = np.load(f"{HERE}/cache/pub_{args.tag}.npz")
    H, W = [int(x) for x in cache["HW"]]
    f8 = LooPool(cache["s8"]).pooled("median")
    fu = upsample(f8, H, W, "cubic")

    files = sorted(f for f in os.listdir(args.render_dir) if f.endswith(".png"))[:args.n]
    print(f"round-trip isolation on {len(files)} production renders {W}x{H}, "
          f"field mean|d| {np.abs(fu).mean():.4f}px max {np.abs(fu).max():.3f}px\n")

    arms = [("cubic", lambda im, f: base_warp(im, f, "cubic")),
            ("lanczos4  [r27]", lambda im, f: base_warp(im, f, "lanczos")),
            ("spline5", lambda im, f: base_warp(im, f, "spline5")),
            ("fft2+cubic", lambda im, f: warp_bandlimited(im, f, 2, "cubic")),
            ("fft2+lanczos", lambda im, f: warp_bandlimited(im, f, 2, "lanczos")),
            ("fft4+cubic", lambda im, f: warp_bandlimited(im, f, 4, "cubic"))]

    acc = {a: [0.0, 0.0, 0.0] for a, _ in arms}
    for fn in files:
        img = np.asarray(Image.open(os.path.join(args.render_dir, fn)).convert("RGB"),
                         dtype=np.float32) / 255.0
        e0 = hf_energy(img)
        for name, fnc in arms:
            t0 = time.time()
            a = np.clip(fnc(img, fu), 0, 1)
            b = np.clip(fnc(a, -fu), 0, 1)
            # score the interior only: the border is where reflect/Gibbs effects live and
            # a round trip there is not informative about the shipped centre of the frame
            m = 32
            d = ((b[m:-m, m:-m] - img[m:-m, m:-m]) ** 2).mean()
            acc[name][0] += 10 * np.log10(1.0 / max(d, 1e-12))
            acc[name][1] += hf_energy(a) / e0          # ONE warp: what ships
            acc[name][2] += time.time() - t0
    n = len(files)
    print(f"{'kernel':>16} {'roundtrip dB':>13} {'HF kept (1 warp)':>18} {'s/img':>8}")
    for name, _ in arms:
        P, E, T = (x / n for x in acc[name])
        print(f"{name:>16} {P:13.2f} {E:18.4f} {T / 2:8.2f}")
    print("\nHF kept is the operative number: 1.000 would be a warp that destroys nothing.")


if __name__ == "__main__":
    main()

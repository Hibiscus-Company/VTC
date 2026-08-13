"""Block phase-correlation field: a different ESTIMATOR of the same global misregistration.

DIS is a coarse-to-fine variational flow tuned for large motion; the thing we are measuring is
a 0.1-1 px static warp.  Phase correlation is exact for pure translation and reads sub-pixel
shift straight off the cross-power-spectrum peak, so it should be the better instrument at this
scale -- and it has an explicit resolution knob (window size / stride) instead of DIS's implicit
one, which is exactly lever (a) done properly.
"""
import os, sys, time
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(8)
OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/pc"
os.makedirs(OUT, exist_ok=True)


def sign_check():
    """Establish phaseCorrelate's sign convention against a known shift."""
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (0, 0), 1.5)
    s = (3.0, -2.0)  # render = photo shifted by s  ->  render(x) = photo(x - s)
    M = np.float32([[1, 0, s[0]], [0, 1, s[1]]])
    b = cv2.warpAffine(a, M, (256, 256), flags=cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_REFLECT)
    win = cv2.createHanningWindow((256, 256), cv2.CV_32F)
    d, _ = cv2.phaseCorrelate(a, b, win)
    print(f"true shift {s}, phaseCorrelate(photo,render) -> {d}")
    return d


def pc_field(rg, gg, win=128, stride=64, maxshift=4.0):
    """grid of sub-pixel shifts; returns (h,w,2) with the fit_field sign convention:
    sampling the render at (x + f) lands on the photo."""
    H, W = rg.shape
    ys = list(range(0, H - win + 1, stride))
    xs = list(range(0, W - win + 1, stride))
    hann = cv2.createHanningWindow((win, win), cv2.CV_32F)
    out = np.zeros((len(ys), len(xs), 2), np.float32)
    conf = np.zeros((len(ys), len(xs)), np.float32)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            g = gg[y:y + win, x:x + win]
            r = rg[y:y + win, x:x + win]
            (dx, dy), c = cv2.phaseCorrelate(g, r, hann)
            if abs(dx) > maxshift or abs(dy) > maxshift:
                dx = dy = 0.0
                c = 0.0
            out[iy, ix] = (dx, dy)
            conf[iy, ix] = c
    return out, conf


def build(name, render_dir, gt_dir, win=128, stride=64):
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)}
    files = sorted(f for f in os.listdir(render_dir) if f.lower().endswith(".png"))
    S, C = [], []
    t0 = time.time()
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by:
            continue
        r = np.asarray(Image.open(os.path.join(render_dir, f)).convert("RGB"), np.float32) / 255
        g = np.asarray(Image.open(os.path.join(gt_dir, gt_by[stem])).convert("RGB"), np.float32) / 255
        if r.shape != g.shape:
            continue
        rg = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
        gg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
        f2, c2 = pc_field(rg, gg, win, stride)
        S.append(f2); C.append(c2)
    dst = f"{OUT}/{name}_w{win}s{stride}.npz"
    np.savez_compressed(dst, stack=np.stack(S), conf=np.stack(C))
    print(f"{name} w{win}s{stride}: {len(S)} pairs {time.time()-t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "signcheck":
        sign_check(); sys.exit()
    scenes = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
    for win, stride in ((256, 64), (128, 32), (64, 16)):
        for s in scenes:
            build(s, f"/mnt/d/avv/output/{s}_gsplatB9ut/train_renders",
                  f"/mnt/d/avv/data/phase1/public_set/{s}/train/images", win, stride)

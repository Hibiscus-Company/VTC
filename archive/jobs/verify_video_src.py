#!/usr/bin/env python
"""Two checks before rebuilding the video scenes for r23.

1) PROVENANCE: does re-encoding the candidate production PNG dir with the SHIPPED settings
   (q100 ss2 optimize progressive) reproduce r21's zip bytes exactly? If yes, that dir is
   provably the true source and we can safely re-encode it differently.
2) CALIBRATION: the +1.0088 bonsai gain was measured on a SINGLE-member eval render. Production
   bonsai is a 3-member pixel-mean (smoother => less high-frequency for JPEG to destroy), so the
   real gain is probably smaller. Measure encode SELF-DISTORTION (LPIPS between the PNG and its
   JPEG round-trip) on both, GT-free, and use the ratio to scale the estimate honestly.
"""
import io, os, sys, zipfile
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None

R21 = "/mnt/d/avv/submissions/sub_round21_chairdepth_hcm0674ema.zip"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
CAND = {"bonsai": "/mnt/d/avv/r14/bonsai_ens/png", "chair": "/mnt/d/avv/r21/video_ens/chair7/png"}


def enc(im, **kw):
    b = io.BytesIO(); im.save(b, "JPEG", **kw); return b.getvalue()


def main():
    import lpips as lpips_pkg
    dev = "cuda"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    z = zipfile.ZipFile(R21)
    names = {n for n in z.namelist()}

    print("=== 1. PROVENANCE: re-encode candidate PNGs with shipped settings vs r21 zip bytes ===")
    for scene, d in CAND.items():
        arc = sorted(n for n in names if n.startswith(scene + "/"))
        ok = bad = miss = 0
        for a in arc:
            stem = os.path.splitext(os.path.basename(a))[0]
            p = os.path.join(d, stem + ".png")
            if not os.path.exists(p):
                miss += 1; continue
            mine = enc(Image.open(p).convert("RGB"), **SHIPPED)
            if mine == z.read(a):
                ok += 1
            else:
                bad += 1
        print(f"  {scene}: {ok} byte-identical, {bad} differ, {miss} missing  (of {len(arc)})")

    print("\n=== 2. CALIBRATION: encode self-distortion (LPIPS png vs jpeg-roundtrip), GT-free ===")
    cases = [
        ("bonsai PRODUCTION (3-member)", "/mnt/d/avv/r14/bonsai_ens/png"),
        ("bonsai EVAL (1-member)", "/mnt/d/avv/tw_test/bonsai_ema099/eval_png"),
        ("chair PRODUCTION (7-member)", "/mnt/d/avv/r21/video_ens/chair7/png"),
        ("chair EVAL (4-member)", "/mnt/d/avv/restore/chair_ens/png"),
    ]
    for label, d in cases:
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))[:20]
        ship = hq = 0.0
        for f in fs:
            im = Image.open(os.path.join(d, f)).convert("RGB")
            ref = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
            for kw, acc in ((SHIPPED, "ship"),
                            (dict(quality=98, subsampling=0, optimize=True,
                                  progressive=True, keep_rgb=True), "hq")):
                dec = Image.open(io.BytesIO(enc(im, **kw))).convert("RGB")
                t = torch.from_numpy(np.asarray(dec, np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
                with torch.no_grad():
                    v = float(vgg(ref * 2 - 1, t * 2 - 1).item())
                if acc == "ship":
                    ship += v
                else:
                    hq += v
        n = len(fs)
        print(f"  {label:32s} n={n:3d}  self-LPIPS shipped {ship/n:.5f}  hq {hq/n:.5f}  "
              f"reduction {100*(1-(hq/n)/max(ship/n,1e-9)):5.1f}%")


if __name__ == "__main__":
    main()

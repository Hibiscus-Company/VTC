#!/usr/bin/env python
"""Driver: score Laplacian-pyramid fusion configs on the PRODUCTION harness (HCM0181, 60 real
test poses, real test GT, models trained on 100% of train photos).

Per-image metrics are cached to JSON so downstream cross-validation (view folds, member folds)
uses exactly the same numbers as the headline table.
"""
import os, sys, io, json, argparse, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lapfuse import lap_pyr, lap_recon, fuse, fuse_image, _K
Image.MAX_IMAGE_PIXELS = None

GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
POOL = {
    "m1": "/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
    "m2": "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
    "m3": "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
    "m4": "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png",
    # secondary family (different configs, same scene / same poses) -- generalisation check
    "s0": "/mnt/d/avv/output/HCM0181_sh0/test_poses_renders_png",
    "s1": "/mnt/d/avv/output/HCM0181_sh1/test_poses_renders_png",
    "s2": "/mnt/d/avv/output/HCM0181_sh2/test_poses_renders_png",
    "s3": "/mnt/d/avv/output/HCM0181_sh3/test_poses_renders_png",
    "b1": "/mnt/d/avv/output/HCM0181_gsplatB1/test_poses_renders_png",
    "b2": "/mnt/d/avv/output/HCM0181_gsplatB2/test_poses_renders_png",
    "b3": "/mnt/d/avv/output/HCM0181_gsplatB3/test_poses_renders_png",
    "b8": "/mnt/d/avv/output/HCM0181_gsplatB8pure/test_poses_renders_png",
}

JPEG_KW = dict(quality=100, subsampling=2, optimize=True, progressive=True)


class Harness:
    def __init__(self, members, device="cuda", nlev=5):
        from utils.loss_utils import ssim as repo_ssim
        import lpips as lpips_pkg
        self.ssim = repo_ssim
        self.dev = device
        self.vgg = lpips_pkg.LPIPS(net="vgg").to(device).eval()
        self.nlev = nlev
        self.k = _K.to(device)
        gtf = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
        dirs = [POOL[m] for m in members]
        self.stems = sorted(s for s in gtf
                            if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
        self.members = members
        # cache uint8 in RAM
        self.M = []   # per stem: uint8 array [k,H,W,3]
        self.G = []
        for s in self.stems:
            self.M.append(np.stack([np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"))
                                    for d in dirs]))
            self.G.append(np.asarray(Image.open(os.path.join(GT, gtf[s])).convert("RGB")))
        print(f"harness: {len(self.stems)} poses x {len(members)} members {members}", flush=True)

    def stack(self, i, sub=None):
        a = self.M[i]
        if sub is not None:
            a = a[sub]
        t = torch.from_numpy(a).to(self.dev).float().permute(0, 3, 1, 2) / 255.0
        return t

    def gt(self, i):
        return (torch.from_numpy(self.G[i]).to(self.dev).float().permute(2, 0, 1).unsqueeze(0) / 255.0)

    def metrics(self, img, i, jpeg=False):
        """img float [1,3,H,W] in [0,1] -> quantise to uint8 (production always writes uint8)."""
        a = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
        if jpeg:
            pil = Image.fromarray(a[0].permute(1, 2, 0).cpu().numpy())
            buf = io.BytesIO(); pil.save(buf, "JPEG", **JPEG_KW); buf.seek(0)
            a = torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"))).to(self.dev)
            a = a.permute(2, 0, 1).unsqueeze(0)
        r = a.float() / 255.0
        g = self.gt(i)
        with torch.no_grad():
            mse = ((r - g) ** 2).mean().item()
            P = 10 * np.log10(1.0 / max(mse, 1e-12))
            S = float(self.ssim(r, g))
            L = float(self.vgg(r * 2 - 1, g * 2 - 1).item())
        return P, S, L

    def score_cfg(self, cfg, sub=None, jpeg=False):
        rows = []
        for i in range(len(self.stems)):
            st = self.stack(i, sub)
            if cfg is None:
                img = st.mean(0, keepdim=True)
            else:
                img = fuse_image(st, cfg, self.nlev, self.k)
            rows.append(self.metrics(img, i, jpeg))
            del st, img
        return np.array(rows)   # [n,3] P,S,L


def agg(rows):
    P, S, L = rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 2].mean()
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), P, S, L

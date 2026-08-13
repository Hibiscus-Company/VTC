"""Shared harness scoring lib for the AGGREGATOR study (HCM0181 production harness)."""
import os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None

GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
OUT = "/mnt/d/avv/output"

_vgg = None
_ssim = None


def init(dev="cuda"):
    global _vgg, _ssim
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    _ssim = repo_ssim
    _vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    return dev


def mdir(v):
    return os.path.join(OUT, "HCM0181_" + v, "test_poses_renders_png")


def stems_for(variants):
    gt_by = {os.path.splitext(f)[0]: os.path.join(GT, f) for f in os.listdir(GT)}
    ds = [mdir(v) for v in variants]
    st = sorted(s for s in gt_by
                if all(os.path.exists(os.path.join(d, s + ".png")) for d in ds))
    return st, gt_by


def load(d, s):
    return np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                      dtype=np.float32) / 255.0


def load_gt(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def metrics(img_np, gt_t, dev):
    """returns (psnr, ssim, lpips_vgg) for one image against a GT tensor."""
    r = torch.from_numpy(np.ascontiguousarray(np.clip(img_np, 0, 1), dtype=np.float32)
                         ).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        mse = ((r - gt_t) ** 2).mean().item()
        p = 10 * np.log10(1.0 / max(mse, 1e-12))
        s = float(_ssim(r, gt_t))
        l = float(_vgg(r * 2 - 1, gt_t * 2 - 1).item())
    return p, s, l


class Scorer:
    """Accumulates PSNR/SSIM/LPIPS per named aggregator."""

    def __init__(self, dev="cuda"):
        self.dev = dev
        self.acc = {}
        self.n = 0

    def add(self, name, img_np, gt_t):
        r = torch.from_numpy(np.clip(img_np, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            mse = ((r - gt_t) ** 2).mean().item()
            p = 10 * np.log10(1.0 / max(mse, 1e-12))
            s = float(_ssim(r, gt_t))
            l = float(_vgg(r * 2 - 1, gt_t * 2 - 1).item())
        a = self.acc.setdefault(name, [0.0, 0.0, 0.0, 0])
        a[0] += p; a[1] += s; a[2] += l; a[3] += 1

    def table(self):
        rows = []
        for k, (P, S, L, n) in self.acc.items():
            P, S, L = P / n, S / n, L / n
            sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
            rows.append((k, sc, P, S, L, n))
        return rows


def gt_tensor(path, dev):
    return torch.from_numpy(load_gt(path)).permute(2, 0, 1).unsqueeze(0).to(dev)

"""Shared harness utilities: load harness image sets, project-scorer metrics."""
import os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
import torch.nn.functional as F
from utils.loss_utils import ssim as repo_ssim, create_window
import lpips as lpips_pkg

DEV = "cuda"
SCENES = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
GT_TEST = "/mnt/d/avv/data/phase1/public_set/{s}/test/images"
GT_TRAIN = "/mnt/d/avv/data/phase1/public_set/{s}/train/images"
R_TEST = "/mnt/d/avv/output/{s}_gsplatB9ut/test_poses_renders_png"
R_TRAIN = "/mnt/d/avv/output/{s}_gsplatB9ut/train_renders"
K4 = "/mnt/d/avv/prodharness/k4/png"

_vgg = None
def getvgg():
    global _vgg
    if _vgg is None:
        _vgg = lpips_pkg.LPIPS(net="vgg").to(DEV).eval()
    return _vgg


def load(p):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)


def pairs(render_dir, gt_dir):
    """List of (stem, render_path, gt_path) for stems present in both."""
    gt_by = {os.path.splitext(f)[0]: os.path.join(gt_dir, f) for f in os.listdir(gt_dir)}
    out = []
    for f in sorted(os.listdir(render_dir)):
        if os.path.splitext(f)[1].lower() not in (".png", ".jpg", ".jpeg"):
            continue
        s = os.path.splitext(f)[0]
        if s in gt_by:
            out.append((s, os.path.join(render_dir, f), gt_by[s]))
    return out


def ssim_map(img1, img2, window_size=11):
    """Full (non-averaged) SSIM map, exactly as the repo computes it (zero-pad conv)."""
    channel = img1.size(-3)
    window = create_window(window_size, channel).type_as(img1).to(img1.device)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))


def lpips_layers(r, g):
    """Per-VGG-scale LPIPS contributions. Returns np array of 5 floats summing to LPIPS."""
    m = getvgg()
    in0 = m.scaling_layer(r * 2 - 1)
    in1 = m.scaling_layer(g * 2 - 1)
    o0, o1 = m.net.forward(in0), m.net.forward(in1)
    res = []
    for kk in range(m.L):
        f0 = lpips_pkg.normalize_tensor(o0[kk])
        f1 = lpips_pkg.normalize_tensor(o1[kk])
        d = (f0 - f1) ** 2
        res.append(float(m.lins[kk](d).mean()))
    return np.array(res)


def score(P, S, L, psnr_max=50.0):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / psnr_max, 1.0))


def metrics(render_dir, gt_dir, transform=None, per_image=False):
    """Returns dict with mean PSNR/SSIM/LPIPS/score. transform(r, g)->r' applied to render."""
    ps, ss, ls = [], [], []
    with torch.no_grad():
        for stem, rp, gp in pairs(render_dir, gt_dir):
            r = load(rp).to(DEV); g = load(gp).to(DEV)
            if transform is not None:
                r = transform(r, g, stem)
            mse = ((r - g) ** 2).mean().item()
            ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
            ss.append(float(repo_ssim(r, g)))
            ls.append(float(getvgg()(r * 2 - 1, g * 2 - 1).item()))
    P, S, L = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
    d = dict(n=len(ps), PSNR=P, SSIM=S, LPIPS=L, SCORE=score(P, S, L))
    if per_image:
        d["psnr_list"] = ps; d["ssim_list"] = ss; d["lpips_list"] = ls
    return d

"""Scoring core for the ensemble-mix lever.

Loads the verified uint8 cache (21 HCM0181 render variants + prebuilt k4, 60 real
test poses, real test GT) and scores arbitrary weighted pixel-mean ensembles with
the PROJECT metric (repo SSIM, lpips-vgg, PSNR=10log10(1/mse)).

Ensembles are rounded to uint8 before scoring -- production writes PNG/JPEG, so the
quantisation is part of the pipeline.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
DEV = os.environ.get("DEV", "cuda:0")

_R = None; _G = None; _NAMES = None


def data():
    global _R, _G, _NAMES
    if _R is None:
        _R = np.load(os.path.join(TMP, "renders_u8.npy"), mmap_mode="r")
        _G = np.load(os.path.join(TMP, "gt_u8.npy"), mmap_mode="r")
        _NAMES = open(os.path.join(TMP, "names.txt")).read().split()
    return _R, _G, _NAMES


_VGG = None
def vgg():
    global _VGG
    if _VGG is None:
        _VGG = lpips_pkg.LPIPS(net="vgg").to(DEV).eval()
    return _VGG


def combine(P, S, L):
    """P,S,L are means over views."""
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))


def score_weights(cands, views=None, verbose=True, extra_imgs=None):
    """cands: dict name -> weight vector (len 21, over the 21 base variants).
    Returns dict name -> {'psnr':[per-view], 'ssim':[...], 'lpips':[...]}.
    extra_imgs: dict name -> uint8 array [nviews,H,W,3] already built (e.g. k4)."""
    R, G, NAMES = data()
    nM = 21
    if views is None:
        views = list(range(R.shape[1]))
    names = list(cands.keys())
    W = torch.tensor(np.stack([cands[n] for n in names]).astype(np.float32), device=DEV)  # [C,21]
    out = {n: {"psnr": [], "ssim": [], "lpips": []} for n in names}
    if extra_imgs:
        for n in extra_imgs:
            out[n] = {"psnr": [], "ssim": [], "lpips": []}
    net = vgg()
    with torch.no_grad():
        for vi in views:
            M = torch.from_numpy(np.ascontiguousarray(R[:nM, vi])).to(DEV)          # [21,H,W,3] u8
            M = M.permute(0, 3, 1, 2).float()                                        # [21,3,H,W] 0..255
            g = torch.from_numpy(np.ascontiguousarray(G[vi])).to(DEV).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            def gen():
                for ci, n in enumerate(names):
                    r = torch.einsum("m,mchw->chw", W[ci], M)
                    yield n, (torch.round(r).clamp_(0, 255).unsqueeze(0) / 255.0)
                if extra_imgs:
                    for n, arr in extra_imgs.items():
                        yield n, (torch.from_numpy(np.ascontiguousarray(arr[vi])).to(DEV)
                                  .permute(2, 0, 1).float().unsqueeze(0) / 255.0)
            for n, r in gen():
                for attempt in range(60):
                    try:
                        mse = float(((r - g) ** 2).mean())
                        sv = float(repo_ssim(r, g))
                        lv = float(net(r * 2 - 1, g * 2 - 1).item())
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); time.sleep(5)
                else:
                    raise RuntimeError("persistent OOM")
                out[n]["psnr"].append(10 * np.log10(1.0 / max(mse, 1e-12)))
                out[n]["ssim"].append(sv)
                out[n]["lpips"].append(lv)
                del r
            del M, g
            torch.cuda.empty_cache()
            if verbose and (vi % 10 == 0):
                print(f"  view {vi} done", flush=True)
    return out


def agg(rec, views=None):
    """rec: {'psnr':[...],...} -> (score, P, S, L)"""
    idx = range(len(rec["psnr"])) if views is None else views
    P = float(np.mean([rec["psnr"][i] for i in idx]))
    S = float(np.mean([rec["ssim"][i] for i in idx]))
    L = float(np.mean([rec["lpips"][i] for i in idx]))
    return combine(P, S, L), P, S, L


def onehot(i, n=21):
    w = np.zeros(n, np.float64); w[i] = 1.0; return w


def uniform(idxs, n=21):
    w = np.zeros(n, np.float64); w[list(idxs)] = 1.0 / len(idxs); return w

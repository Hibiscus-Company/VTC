#!/usr/bin/env python
"""LEVER: frequency-domain / multi-scale (Laplacian-pyramid) fusion of ensemble members.

Baseline = pixel mean of k members.  Because the Burt-Adelson Laplacian pyramid is an exact,
LINEAR analysis/synthesis pair, using the 'mean' rule at every level reproduces the pixel mean
BIT-EXACTLY (asserted at startup).  Any deviation therefore comes purely from the alternative
per-level combination rule, which is the thing under test.

Rules available at each Laplacian level:
  mean        L_bar                                (identity -> pixel mean)
  maxmag      per-pixel pick the member with max local |L| (classic multi-focus fusion)
  median      per-pixel median across members
  pnorm       salience-weighted: w_i ~ (localE_i)^p  (p=0 -> mean, p->inf -> maxmag)
  gain        g * L_bar                             (per-level unsharp / attenuation)
  inject      L_bar + lam*(L_j - L_bar)             (pyramid texture-injection, member j)
  energy      L_bar * (1 + lam*(r-1)), r = sqrt(mean_i localE_i / localE(L_bar))
              -> restores the local HF ENERGY that averaging destroys, keeps the mean's structure
"""
import os, sys, io, json, argparse, itertools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None

_K = torch.tensor([1., 4., 6., 4., 1.]) / 16.


def _blur(x, k):
    """x [B,C,H,W] separable 5-tap binomial, reflect pad."""
    C = x.shape[1]
    kr = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    kc = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), kr, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), kc, groups=C)
    return x


def pyr_down(x, k):
    return _blur(x, k)[:, :, ::2, ::2]


def pyr_up(x, size, k):
    B, C, H, W = x.shape
    up = x.new_zeros(B, C, H * 2, W * 2)
    up[:, :, ::2, ::2] = x
    up = _blur(up, k * 2.0)
    return up[:, :, :size[0], :size[1]]


def lap_pyr(x, n, k):
    """returns (laps[0..n-1] fine->coarse, residual)"""
    laps, sizes, g = [], [], x
    for _ in range(n):
        sizes.append(g.shape[-2:])
        d = pyr_down(g, k)
        laps.append(g - pyr_up(d, g.shape[-2:], k))
        g = d
    return laps, g, sizes


def lap_recon(laps, res, sizes, k):
    g = res
    for i in range(len(laps) - 1, -1, -1):
        g = laps[i] + pyr_up(g, sizes[i], k)
    return g


def boxf(x, w):
    if w <= 1:
        return x
    p = w // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), w, stride=1)


# ---------------------------------------------------------------- fusion rules
def fuse(Ls, rule, **kw):
    """Ls: [k,C,H,W] laplacian coefficients of the k members at one level."""
    kk = Ls.shape[0]
    mean = Ls.mean(0, keepdim=True)
    if rule == "mean":
        return mean
    if rule == "median":
        s, _ = torch.sort(Ls, dim=0)
        if kk % 2 == 1:
            return s[kk // 2:kk // 2 + 1]
        return 0.5 * (s[kk // 2 - 1:kk // 2] + s[kk // 2:kk // 2 + 1])
    if rule == "maxmag":
        w = kw.get("win", 3)
        E = boxf((Ls ** 2).sum(1, keepdim=True), w)          # [k,1,H,W] local energy (RGB summed)
        idx = E.argmax(0, keepdim=True)                       # [1,1,H,W]
        return torch.gather(Ls, 0, idx.expand(-1, Ls.shape[1], -1, -1))
    if rule == "pnorm":
        p, w = kw.get("p", 2.0), kw.get("win", 3)
        E = boxf((Ls ** 2).sum(1, keepdim=True), w)
        wts = (E + 1e-12) ** (p / 2.0)
        wts = wts / wts.sum(0, keepdim=True)
        return (wts * Ls).sum(0, keepdim=True)
    if rule == "gain":
        return kw.get("g", 1.0) * mean
    if rule == "inject":
        j, lam = kw.get("j", 0), kw.get("lam", 0.5)
        return mean + lam * (Ls[j:j + 1] - mean)
    if rule == "energy":
        lam, w = kw.get("lam", 1.0), kw.get("win", 5)
        Em = boxf((Ls ** 2).sum(1, keepdim=True), w).mean(0, keepdim=True)   # avg member energy
        Eb = boxf((mean ** 2).sum(1, keepdim=True), w)                        # mean's energy
        r = torch.sqrt((Em + 1e-10) / (Eb + 1e-10))
        r = r.clamp(max=kw.get("rmax", 4.0))
        # --- CONTROLS: keep the same r DISTRIBUTION but break its spatial correspondence with
        #     where the members actually disagree.  If these match the real map, the "ensemble
        #     disagreement" information is doing nothing and this is just adaptive sharpening.
        m = kw.get("map", "raw")
        if m == "const":                       # per-image scalar = mean of r
            r = r.mean().expand_as(r)
        elif m == "shuffle":                   # same histogram, scrambled location
            flat = r.reshape(-1)
            r = flat[torch.randperm(flat.numel(), device=r.device)].reshape(r.shape)
        elif m.startswith("blur"):             # spatially smoothed map
            r = boxf(r, int(m[4:]))
        return mean * (1.0 + lam * (r - 1.0))
    raise ValueError(rule)


def fuse_image(stack, cfg, nlev, k):
    """stack [k,3,H,W] members. cfg = dict(nalt=int, rule=str, **kw). Returns [1,3,H,W]."""
    laps, res, sizes = lap_pyr(stack, nlev, k)
    out_l = []
    for i, L in enumerate(laps):
        if i < cfg["nalt"]:
            out_l.append(fuse(L, cfg["rule"], **cfg.get("kw", {})))
        else:
            out_l.append(L.mean(0, keepdim=True))
    return lap_recon(out_l, res.mean(0, keepdim=True), sizes, k)

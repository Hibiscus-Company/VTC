#!/usr/bin/env python
"""H-B BOUND TEST (strategy audit 23/07, rank-1): can per-frame blur/appearance MATCHING move
a video scene's score? All prior "blur ceiling" kills (eps2d, B3, texture_weight) pushed in
the SHARPENING direction; this tests the MATCHING direction -- fit a tiny per-frame transform
that maps render -> photo on TRAIN pairs, interpolate its params by frame index to the eval
holes, apply to eval renders, score. If the bound is flat, H-B (and the blur-kernel moonshot)
dies for ~zero GPU cost. If it's alive, productionize as render-space post-processing.

Per-frame model (5 params): out = gain * [ (1+a)*I - a*(I * G(sx,sy,th)) ] + bias
  a > 0: unsharp (GT sharper than render)   a < 0: blur (GT blurrier)   a=0: identity
  G = anisotropic Gaussian kernel (sx, sy, theta), K x K, depthwise conv, reflect pad.

RULE 10: fit uses TRAIN photos only. Transfer to eval poses uses FRAME INDEX only (same
precedent as the validated appearance-interpolation finding, EXPERIMENTS.md round-2 EDA #6).
Eval GT is held-out TRAIN photos (standard eval-split protocol), read only by eval_score.
"""
import argparse, csv, json, math, os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = None


def load(p, dev):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(dev)


def gauss_kernel(sx, sy, th, K, dev):
    r = K // 2
    y, x = torch.meshgrid(torch.arange(-r, r + 1, device=dev, dtype=torch.float32),
                          torch.arange(-r, r + 1, device=dev, dtype=torch.float32),
                          indexing="ij")
    ct, st = torch.cos(th), torch.sin(th)
    xr = ct * x + st * y
    yr = -st * x + ct * y
    k = torch.exp(-0.5 * ((xr / sx.clamp(min=0.1)) ** 2 + (yr / sy.clamp(min=0.1)) ** 2))
    return (k / k.sum()).view(1, 1, K, K)


def apply_model(img, prm, K):
    # prm: dict of scalars (tensors); img [1,3,H,W]
    dev = img.device
    k = gauss_kernel(prm["sx"], prm["sy"], prm["th"], K, dev).expand(3, 1, K, K)
    r = K // 2
    blur = F.conv2d(F.pad(img, (r, r, r, r), mode="reflect"), k, groups=3)
    out = prm["gain"] * ((1 + prm["a"]) * img - prm["a"] * blur) + prm["bias"]
    return out.clamp(0, 1)


def fit_frame(render, photo, K, iters, lr, dev, gain_only=False):
    prm = {
        "sx": torch.tensor(1.0, device=dev, requires_grad=not gain_only),
        "sy": torch.tensor(1.0, device=dev, requires_grad=not gain_only),
        "th": torch.tensor(0.0, device=dev, requires_grad=not gain_only),
        "a": torch.tensor(0.0, device=dev, requires_grad=not gain_only),
        "gain": torch.tensor(1.0, device=dev, requires_grad=True),
        "bias": torch.tensor(0.0, device=dev, requires_grad=True),
    }
    opt = torch.optim.Adam([p for p in prm.values() if p.requires_grad], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        out = prm["gain"] * ((1 + prm["a"]) * render - prm["a"] * F.conv2d(
            F.pad(render, (K // 2,) * 4, mode="reflect"),
            gauss_kernel(prm["sx"], prm["sy"], prm["th"], K, dev).expand(3, 1, K, K),
            groups=3)) + prm["bias"]
        loss = (out.clamp(0, 1) - photo).abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = apply_model(render, prm, K)
        final = (out - photo).abs().mean().item()
        ident = (render - photo).abs().mean().item()
    return {n: float(p.detach()) for n, p in prm.items()}, final, ident


def fidx(name):
    return int(os.path.splitext(name)[0].split("_")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_render", required=True, help="renders at train_sub poses")
    ap.add_argument("--train_gt", required=True, help="train_sub photos dir")
    ap.add_argument("--eval_render", required=True, help="same model's eval-hole renders")
    ap.add_argument("--out", required=True, help="corrected eval renders (png)")
    ap.add_argument("--params_json", required=True)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gain_only", action="store_true",
                    help="ablation: exposure-only (a/sx/sy/th frozen at identity)")
    args = ap.parse_args()
    dev = "cuda"

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.train_gt)}
    fits = {}
    tr_files = sorted(os.listdir(args.train_render))
    for i, f in enumerate(tr_files):
        s = os.path.splitext(f)[0]
        assert s in gt_by, f"train render {f} lacks GT"
        r = load(os.path.join(args.train_render, f), dev)
        g = load(os.path.join(args.train_gt, gt_by[s]), dev)
        assert r.shape == g.shape, f"shape mismatch {f}"
        prm, final, ident = fit_frame(r, g, args.K, args.iters, args.lr, dev,
                                      gain_only=args.gain_only)
        fits[fidx(f)] = {"prm": prm, "l1_after": final, "l1_before": ident}
        if i % 25 == 0:
            print(f"[fit {i}/{len(tr_files)}] {s}: L1 {ident:.4f} -> {final:.4f}  {prm}")

    idxs = sorted(fits.keys())
    med_gain = float(np.median([fits[i]["prm"]["gain"] for i in idxs]))
    med_a = float(np.median([fits[i]["prm"]["a"] for i in idxs]))
    print(f"fitted {len(idxs)} train frames; median gain {med_gain:.4f} a {med_a:.4f}")

    os.makedirs(args.out, exist_ok=True)
    interp_log = {}
    with torch.no_grad():
        for f in sorted(os.listdir(args.eval_render)):
            if not f.lower().endswith((".png", ".jpg")):
                continue
            ei = fidx(f)
            lo = max([i for i in idxs if i <= ei], default=None)
            hi = min([i for i in idxs if i >= ei], default=None)
            if lo is None:
                lo = hi
            if hi is None:
                hi = lo
            w = 0.5 if lo == hi else (ei - lo) / (hi - lo)
            prm = {n: torch.tensor((1 - w) * fits[lo]["prm"][n] + w * fits[hi]["prm"][n],
                                   device=dev)
                   for n in fits[lo]["prm"]}
            img = load(os.path.join(args.eval_render, f), dev)
            out = apply_model(img, prm, args.K)
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(args.out, os.path.splitext(f)[0] + ".png"))
            interp_log[f] = {n: float(prm[n]) for n in prm}

    with open(args.params_json, "w") as f:
        json.dump({"train_fits": {str(k): v for k, v in fits.items()},
                   "eval_interp": interp_log, "gain_only": args.gain_only,
                   "K": args.K, "iters": args.iters}, f, indent=2)
    print(f"wrote {len(interp_log)} corrected eval renders -> {args.out}")


if __name__ == "__main__":
    main()

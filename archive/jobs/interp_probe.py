#!/usr/bin/env python
"""GAME-CHANGER PROBE (strategy audit lever 3): classical-flow frame interpolation on the
video scenes' eval holes. Each eval hole sits between two train-sub photos one stride away
(3-4 deg). Interpolate the hole from its flanks (DIS flow, symmetric splat at index-fraction
alpha), score with the competition metric vs the held-out train photo. NO training, legal
(train photos only). Baselines: nearest-flank copy (photo-reuse), and the 3DGS render score
for the same holes (from the eval-selection runs).
"""
import os, sys, json
import numpy as np
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

def fidx(n): return int(os.path.splitext(n)[0].split("_")[1])

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

def interp(a, b, alpha, dis):
    """warp a->mid and b->mid with DIS flow, blend by (1-alpha, alpha)"""
    ga = (cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
    gb = (cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
    fab = dis.calc(ga, gb, None)   # a(p) ~ b(p + fab(p))
    fba = dis.calc(gb, ga, None)
    H, W, _ = a.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # sample a at p + alpha*f_ba? standard backward warp to the midpoint:
    wa = cv2.remap(a, xx + alpha * fba[..., 0], yy + alpha * fba[..., 1],
                   cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    wb = cv2.remap(b, xx + (1 - alpha) * fab[..., 0], yy + (1 - alpha) * fab[..., 1],
                   cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip((1 - alpha) * wa + alpha * wb, 0, 1)

def main():
    dev = "cuda"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for scene in ("chair", "bonsai"):
        ES = f"/mnt/d/avv/evalsplit/{scene}"
        sub = sorted(os.listdir(f"{ES}/train_sub/images"))
        ev = sorted(os.listdir(f"{ES}/eval_gt"))
        sub_idx = {fidx(n): n for n in sub}
        st = json.load(open(f"{ES}/split.json"))["stride"]
        out = f"{ES}/interp_probe"; os.makedirs(out, exist_ok=True)
        accs = {"interp": [0, 0, 0], "nearest": [0, 0, 0]}
        n = 0
        with torch.no_grad():
            for name in ev:
                i = fidx(name)
                lo, hi = sub_idx.get(i - st), sub_idx.get(i + st)
                if lo is None or hi is None:
                    continue
                a = load(f"{ES}/train_sub/images/{lo}")
                b = load(f"{ES}/train_sub/images/{hi}")
                g = load(f"{ES}/eval_gt/{name}")
                m = interp(a, b, 0.5, dis)
                Image.fromarray((m * 255 + .5).astype(np.uint8)).save(
                    os.path.join(out, os.path.splitext(name)[0] + ".png"))
                for tag, img in (("interp", m), ("nearest", a)):
                    r = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(dev)
                    gt = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(dev)
                    mse = float(((r - gt) ** 2).mean())
                    accs[tag][0] += 10 * np.log10(1 / max(mse, 1e-12))
                    accs[tag][1] += float(repo_ssim(r, gt))
                    accs[tag][2] += float(vgg(r * 2 - 1, gt * 2 - 1))
                n += 1
        for tag, (P, S, L) in accs.items():
            P, S, L = P / n, S / n, L / n
            sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1))
            print(f"INTERP {scene:7s} {tag:8s} n={n:3d} PSNR {P:7.4f} SSIM {S:.4f} "
                  f"LPIPSvgg {L:.4f} SCORE {sc:.4f}")

if __name__ == "__main__":
    main()

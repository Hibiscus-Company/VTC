#!/usr/bin/env python
"""PER-FRAME blur matching for the motion/defocus-blurred video scenes.

The existing blur_bound test fitted ONE GLOBAL kernel render->photo. With chair's
7.6x within-scene sharpness spread that fit gets inconsistent supervision and is
structurally forced to read flat. This asks the different question:

  does the SCORE-OPTIMAL blur sigma vary materially FROM FRAME TO FRAME,
  and is a frame's optimum predictable from its TRAIN NEIGHBOURS (legal: index only)?

Reports per-frame optimal sigma, the score at sigma=0 vs per-frame-oracle sigma vs
one global sigma, plus a GT-free sharpness proxy of each frame's train neighbours.
"""
import os, sys, argparse
import numpy as np, cv2, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

ap = argparse.ArgumentParser()
ap.add_argument("--render", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--train_dir", default=None, help="train photos, for the neighbour proxy")
ap.add_argument("--sigmas", type=float, nargs="+",
                default=[0.0, 0.25, 0.4, 0.55, 0.7, 0.9, 1.2, 1.6])
ap.add_argument("--tag", default="")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def t(x):
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(dev)


def score(r, g):
    with torch.no_grad():
        R, G = t(r), t(g)
        mse = float(((R - G) ** 2).mean())
        P = 10 * np.log10(1.0 / max(mse, 1e-12))
        S = float(repo_ssim(R, G))
        L = float(vgg(R * 2 - 1, G * 2 - 1))
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), P, S, L


def lapvar(x):
    g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


gtmap = {os.path.splitext(f)[0]: os.path.join(a.gt, f) for f in os.listdir(a.gt)}
rmap = {os.path.splitext(f)[0]: os.path.join(a.render, f) for f in os.listdir(a.render)
        if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
stems = sorted(set(gtmap) & set(rmap))
print(f"{a.tag}: {len(stems)} eval frames, sigmas {a.sigmas}")

tab = np.zeros((len(stems), len(a.sigmas)))
gt_sharp, rn_sharp = [], []
for i, s in enumerate(stems):
    g = load(gtmap[s]); r = load(rmap[s])
    if g.shape != r.shape:
        print("skip", s); continue
    gt_sharp.append(lapvar(g)); rn_sharp.append(lapvar(r))
    for j, sg in enumerate(a.sigmas):
        rr = r if sg == 0 else cv2.GaussianBlur(r, (0, 0), sg)
        tab[i, j] = score(rr, g)[0]

best = tab.argmax(1)
sig = np.array(a.sigmas)
print(f"\n{'frame':>22} {'sig0':>8} {'best_sig':>9} {'best':>8} {'gain':>7} "
      f"{'GTsharp':>9} {'Rsharp':>9}")
for i, s in enumerate(stems[:len(tab)]):
    print(f"{s:>22} {tab[i,0]:8.3f} {sig[best[i]]:9.2f} {tab[i,best[i]]:8.3f} "
          f"{tab[i,best[i]]-tab[i,0]:7.3f} {gt_sharp[i]:9.1f} {rn_sharp[i]:9.1f}")

print(f"\nMEAN score at sigma=0            {tab[:,0].mean():.4f}")
gi = tab.mean(0).argmax()
print(f"MEAN score at BEST GLOBAL sigma  {tab[:,gi].mean():.4f}   (sigma={sig[gi]:.2f}, "
      f"delta {tab[:,gi].mean()-tab[:,0].mean():+.4f})   <- what blur_bound could see")
print(f"MEAN score at PER-FRAME ORACLE   {tab[np.arange(len(tab)),best].mean():.4f}   "
      f"(delta {tab[np.arange(len(tab)),best].mean()-tab[:,0].mean():+.4f})  <- the ceiling")
print(f"per-frame optimal sigma: mean {sig[best].mean():.3f} sd {sig[best].std():.3f} "
      f"range [{sig[best].min():.2f},{sig[best].max():.2f}]")
gs, rs = np.array(gt_sharp), np.array(rn_sharp)
if len(gs) > 3:
    print(f"corr(optimal sigma, GT sharpness)     r = {np.corrcoef(sig[best], gs)[0,1]:+.3f}")
    print(f"corr(optimal sigma, RENDER sharpness) r = {np.corrcoef(sig[best], rs)[0,1]:+.3f}"
          f"   <- render-side => usable at TEST poses with no GT")
    print(f"corr(GT sharpness, RENDER sharpness)  r = {np.corrcoef(gs, rs)[0,1]:+.3f}")
print("\nmean score per sigma:", " ".join(f"{s:.2f}:{v:.3f}" for s, v in zip(sig, tab.mean(0))))

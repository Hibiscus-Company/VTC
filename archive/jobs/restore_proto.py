#!/usr/bin/env python
"""Learned restoration head on top of the 3DGS ensemble render -- HONEST prototype.

Organizer allows any model that renders PNG, so a per-scene image->image restorer is legal.
Rule 10 is still respected: it is trained ONLY on TRAIN photos (the eval-split holdout images
are held-out TRAIN photos, never test GT), no external imagery of these scenes, no manual edits.

THE REGIME POINT (why this is not the usual overfit trap): a restorer trained on renders at
TRAIN poses would learn to fix a render that is already near-perfect (train fit ~27dB) and would
not transfer to novel views. Here every input is a render at a HELD-OUT pose, so the degradation
it learns to undo is genuine novel-view degradation -- the same thing present at test poses.

HONEST PROTOCOL: the held-out poses are themselves split fit/test. The net never sees the test
poses' GT. We report ensemble-only vs restored on those test poses only.
"""
import argparse, os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


class UNet(nn.Module):
    """Small residual U-Net. Predicts a RESIDUAL added to the input render, and is
    zero-initialised at the output so it starts as exact identity -- a restorer that
    cannot help degrades gracefully toward doing nothing instead of destroying the image."""

    def __init__(self, ch=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        self.e1, self.e2, self.e3 = blk(3, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4)
        self.d2 = blk(ch * 4 + ch * 2, ch * 2)
        self.d1 = blk(ch * 2 + ch, ch)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        return x + self.out(d1)


def score(pred, gt, lp):
    """competition score, computed with the PROJECT'S OWN scorer (utils.loss_utils.ssim,
    zero-pad conv) so these numbers are directly comparable to every other measurement in
    this campaign -- a different SSIM implementation would silently shift the baseline."""
    sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
    from utils.loss_utils import ssim as repo_ssim
    ps, ss, ls = [], [], []
    with torch.no_grad():
        for p, g in zip(pred, gt):
            r = torch.from_numpy(np.clip(p, 0, 1)).permute(2, 0, 1).unsqueeze(0).cuda()
            t = torch.from_numpy(np.clip(g, 0, 1)).permute(2, 0, 1).unsqueeze(0).cuda()
            ps.append(10 * np.log10(1.0 / max(((r - t) ** 2).mean().item(), 1e-12)))
            ss.append(float(repo_ssim(r, t)))
            ls.append(float(lp(r * 2 - 1, t * 2 - 1).item()))
    P, S, L = np.mean(ps), np.mean(ss), np.mean(ls)
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), P, S, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True, help="ensemble renders at held-out poses")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--n_fit", type=int, default=40, help="how many poses the restorer may fit on")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_lpips", type=float, default=0.5)
    ap.add_argument("--tag", default="restore")
    ap.add_argument("--seed", type=int, default=0,
                    help="audit 26/07: weight-init and crop RNGs were UNSEEDED, so runs were not "
                         "reproducible and the training-seed variance component read as zero")
    ap.add_argument("--save_dir", default=None,
                    help="audit 26/07: the restored renders were never written, so not a single "
                         "per-image delta could be recomputed. Save them.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    import lpips
    lp = lpips.LPIPS(net="vgg").to(dev)
    for q in lp.parameters():
        q.requires_grad_(False)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.gt_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    def find(d, s):
        for e in (".png", ".jpg", ".JPG", ".jpeg"):
            if os.path.exists(os.path.join(d, s + e)):
                return os.path.join(d, s + e)
        raise FileNotFoundError(f"{s} in {d}")

    R = [load(find(args.render_dir, s)) for s in stems]
    G = [load(find(args.gt_dir, s)) for s in stems]
    print(f"{len(R)} pose pairs, image {R[0].shape}")

    # deterministic split: the restorer NEVER sees test poses
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(R))
    fit, test = idx[:args.n_fit], idx[args.n_fit:]
    print(f"fit on {len(fit)} poses, HELD-OUT test on {len(test)} poses")

    base, bP, bS, bL = score([R[i] for i in test], [G[i] for i in test], lp)
    print(f"BASELINE (ensemble only, test poses): SCORE {base:.4f}  PSNR {bP:.4f} SSIM {bS:.4f} LPIPS {bL:.4f}")

    net = UNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    Rt = [torch.from_numpy(R[i]).permute(2, 0, 1) for i in fit]
    Gt = [torch.from_numpy(G[i]).permute(2, 0, 1) for i in fit]
    H, W = Rt[0].shape[-2:]
    c = min(args.crop, H, W)

    for it in range(args.iters):
        xs, ys = [], []
        for _ in range(args.batch):
            k = np.random.randint(len(Rt))
            i0 = np.random.randint(0, H - c + 1); j0 = np.random.randint(0, W - c + 1)
            xs.append(Rt[k][:, i0:i0 + c, j0:j0 + c])
            ys.append(Gt[k][:, i0:i0 + c, j0:j0 + c])
        x = torch.stack(xs).to(dev); y = torch.stack(ys).to(dev)
        p = net(x)
        loss = F.l1_loss(p, y)
        if args.lambda_lpips > 0:
            loss = loss + args.lambda_lpips * lp(p.clamp(0, 1) * 2 - 1, y * 2 - 1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"  [{it}] loss {loss.item():.4f}")

    net.eval()
    out = []
    with torch.no_grad():
        for i in test:
            x = torch.from_numpy(R[i]).permute(2, 0, 1)[None].to(dev)
            # pad to /4 for the two pooling levels
            ph, pw = (-x.shape[-2]) % 4, (-x.shape[-1]) % 4
            xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
            o = net(xp)[..., :x.shape[-2], :x.shape[-1]]
            out.append(o[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
    new, nP, nS, nL = score(out, [G[i] for i in test], lp)
    print(f"RESTORED (test poses):                SCORE {new:.4f}  PSNR {nP:.4f} SSIM {nS:.4f} LPIPS {nL:.4f}")
    print(f"=== {args.tag}: DELTA {new - base:+.4f} (baseline {base:.4f} -> {new:.4f}) ===")

    # per-image deltas: sign agreement + worst-case jackknife, so the result is auditable
    per = []
    for j, i in enumerate(test):
        b1, _, _, _ = score([R[i]], [G[i]], lp)
        n1, _, _, _ = score([out[j]], [G[i]], lp)
        per.append(n1 - b1)
    per = np.array(per)
    pos = int((per > 0).sum())
    jk = [(per.sum() - per[k]) / (len(per) - 1) for k in range(len(per))]
    print(f"  per-image: {pos}/{len(per)} improved, mean {per.mean():+.4f} sd {per.std(ddof=1):.4f}, "
          f"jackknife range [{min(jk):+.4f}, {max(jk):+.4f}]")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for j, i in enumerate(test):
            Image.fromarray((np.clip(out[j], 0, 1) * 255 + 0.5).astype(np.uint8)).save(
                os.path.join(args.save_dir, stems[i] + ".png"))
        print(f"  saved {len(test)} restored renders -> {args.save_dir}")


if __name__ == "__main__":
    main()

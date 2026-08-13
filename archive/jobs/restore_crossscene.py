#!/usr/bin/env python
"""THE decisive restoration test, in production regime, with the two objections that remain.

Restoration just scored +0.3128 on PRODUCTION-quality renders (models trained on 240/240, real test
poses, real test GT) -- which REFUTES the data-starvation objection that benched it. Two questions
are left, and both are answerable with public-set data we already have:

  Q1 CROSS-SCENE: production restoration for private_set2 cannot be trained on set2 test GT (that is
     the graded set -- Rule 10). So a shipped restorer must either train on set2's held-out TRAIN
     photos (starved, over-corrects) or be trained on one scene and applied to another. Train on
     HCM0181, apply to HCM0193 / HCM0204 / hcm0034. Same rig, different scene. Does it transfer?

  Q2 SURVIVES THE ENCODE: the gain is 96% a single LPIPS-vgg movement, and we just measured that at
     production quality JPEG's structured artifacts are LOAD-BEARING (shipped q100ss2 beats lossless
     PNG). A restorer that "cleans" the image could be destroying exactly the texture JPEG was
     supplying. So score every arm BOTH as PNG and after the shipped JPEG round-trip.

If it transfers cross-scene AND survives the encode, restoration ships. If either fails, it stays
benched -- and this time on production-regime evidence rather than extrapolation.
"""
import io, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from restore_proto import UNet
Image.MAX_IMAGE_PIXELS = None

PUB = "/mnt/d/avv/data/phase1/public_set"
TRAIN_SCENE = "HCM0181"
TRAIN_RENDER = "/mnt/d/avv/prodharness/k4/png"
TEST_SCENES = ["HCM0193", "HCM0204", "hcm0034"]
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    lp = lpips_pkg.LPIPS(net="vgg").to(dev)
    for q in lp.parameters():
        q.requires_grad_(False)

    # ---- train the restorer on HCM0181 production renders, ALL 60 poses ----
    gtd = os.path.join(PUB, TRAIN_SCENE, "test", "images")
    gt_by = {os.path.splitext(f)[0]: os.path.join(gtd, f) for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by if os.path.exists(os.path.join(TRAIN_RENDER, s + ".png")))
    R = [torch.from_numpy(load(os.path.join(TRAIN_RENDER, s + ".png"))).permute(2, 0, 1) for s in stems]
    G = [torch.from_numpy(load(gt_by[s])).permute(2, 0, 1) for s in stems]
    print(f"training restorer on {TRAIN_SCENE}: {len(R)} production test-pose pairs", flush=True)

    net = UNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    H, W = R[0].shape[-2:]; c = min(256, H, W)
    for it in range(3000):
        xs, ys = [], []
        for _ in range(8):
            k = np.random.randint(len(R))
            i0 = np.random.randint(0, H - c + 1); j0 = np.random.randint(0, W - c + 1)
            xs.append(R[k][:, i0:i0 + c, j0:j0 + c]); ys.append(G[k][:, i0:i0 + c, j0:j0 + c])
        x = torch.stack(xs).to(dev); y = torch.stack(ys).to(dev)
        p = net(x)
        loss = F.l1_loss(p, y) + 0.5 * lp(p.clamp(0, 1) * 2 - 1, y * 2 - 1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f"  [{it}] {loss.item():.4f}", flush=True)
    net.eval()

    def score(img, g):
        r = torch.from_numpy(np.clip(img, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(dev)
        t = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P = 10 * np.log10(1.0 / max(((r - t) ** 2).mean().item(), 1e-12))
            S = float(repo_ssim(r, t)); L = float(lp(r * 2 - 1, t * 2 - 1).item())
        return P, S, L

    def jpg(img):
        b = io.BytesIO()
        Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
        b.seek(0)
        return np.asarray(Image.open(b).convert("RGB"), dtype=np.float32) / 255.0

    print(f"\n{'scene':>9} {'arm':>18} {'SCORE':>9} {'LPIPS':>8} {'SSIM':>7} {'PSNR':>8}  {'delta':>8}")
    for sc_name in TEST_SCENES:
        rd = f"/mnt/d/avv/output/{sc_name}_gsplatB9ut/test_poses_renders_png"
        gd = os.path.join(PUB, sc_name, "test", "images")
        if not os.path.isdir(rd):
            print(f"{sc_name}: no renders, skip"); continue
        gb = {os.path.splitext(f)[0]: os.path.join(gd, f) for f in os.listdir(gd)}
        ss = sorted(s for s in gb if os.path.exists(os.path.join(rd, s + ".png")))
        acc = {k: [0.0, 0.0, 0.0] for k in ("base_png", "rest_png", "base_jpg", "rest_jpg")}
        for s in ss:
            img = load(os.path.join(rd, s + ".png")); g = load(gb[s])
            with torch.no_grad():
                x = torch.from_numpy(img).permute(2, 0, 1)[None].to(dev)
                ph, pw = (-x.shape[-2]) % 4, (-x.shape[-1]) % 4
                o = net(F.pad(x, (0, pw, 0, ph), mode="reflect"))[..., :x.shape[-2], :x.shape[-1]]
                out = o[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            for k, im in (("base_png", img), ("rest_png", out),
                          ("base_jpg", jpg(img)), ("rest_jpg", jpg(out))):
                P, S, L = score(im, g)
                acc[k][0] += P; acc[k][1] += S; acc[k][2] += L
        n = len(ss)
        res = {}
        for k in acc:
            P, S, L = (v / n for v in acc[k])
            res[k] = (100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), L, S, P)
        for k in ("base_png", "rest_png", "base_jpg", "rest_jpg"):
            sc_, L, S, P = res[k]
            base = res["base_png"][0] if "png" in k else res["base_jpg"][0]
            print(f"{sc_name:>9} {k:>18} {sc_:9.4f} {L:8.4f} {S:7.4f} {P:8.4f}  {sc_-base:+8.4f}")
        print(f"{'':>9} {'-> PNG delta':>18} {res['rest_png'][0]-res['base_png'][0]:+8.4f}"
              f"   {'JPG delta':>12} {res['rest_jpg'][0]-res['base_jpg'][0]:+8.4f}\n")


if __name__ == "__main__":
    main()

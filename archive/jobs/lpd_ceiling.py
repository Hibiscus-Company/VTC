#!/usr/bin/env python
"""LPIPS-DIRECT diagnostic C: what fraction of the LPIPS budget actually SITS in the
high-disagreement tail?  This bounds every tail-targeted operator (incl. lpd_sub.py).

Spatial LPIPS (per-layer, upsampled to full res) of the PRODUCTION output vs real test GT,
integrated over bands of the shipped r-map.  CPU only -- the GPUs are training.
"""
import io, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, boxf, _K            # noqa: E402
from fieldlib import LooPool, upsample, warp     # noqa: E402
import lpips as L                                 # noqa: E402

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(8)
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
C_SHIP = 4.0 / 3.0
BINS = [1.0, 1.02, 1.05, 1.10, 1.20, 1.50, 1e9]
NAMES = ["r<1.02", "1.02-1.05", "1.05-1.10", "1.10-1.20", "1.20-1.50", "r>1.50"]
LAYERS = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]


def u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def spatial(mS, a, b):
    i0, i1 = mS.scaling_layer(a * 2 - 1), mS.scaling_layer(b * 2 - 1)
    f0, f1 = mS.net.forward(i0), mS.net.forward(i1)
    out = []
    for k in range(mS.L):
        d = (L.normalize_tensor(f0[k]) - L.normalize_tensor(f1[k])) ** 2
        v = mS.lins[k](d)
        out.append(torch.nn.functional.interpolate(v, size=a.shape[-2:], mode="bilinear",
                                                   align_corners=False)[0, 0])
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:n]
    cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
    K = _K
    mS = L.LPIPS(net="vgg", spatial=True, verbose=False).eval()
    NB = len(NAMES)
    lp = np.zeros((5, NB)); npx = np.zeros(NB)
    t0 = time.time()
    for ci, s in enumerate(stems):
        mem = [to_t(u8(os.path.join(d, s + ".png"))) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = to_t(u8(os.path.join(GTD, gt_by[s])))
        L0 = lap_pyr(ens, 1, K)[0][0]
        Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
        V = sum(boxf(((lap_pyr(m, 1, K)[0][0] - L0) ** 2).sum(1, keepdim=True), 3)
                for m in mem) / len(mem)
        r = torch.sqrt(1.0 + C_SHIP * V / (Eb + 1e-10)).clamp(max=4.0)
        x = (ens + (r - 1.0) * L0).clamp(0, 1)
        xn = np.clip(warp(np.ascontiguousarray(x[0].permute(1, 2, 0).numpy()), lens, "lanczos"),
                     0, 1)
        b = io.BytesIO()
        Image.fromarray((xn * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
        xj = to_t(np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.uint8))
        maps = spatial(mS, xj, g)
        rr = r[0, 0]
        for j in range(NB):
            m = (rr >= BINS[j]) & (rr < BINS[j + 1])
            npx[j] += float(m.sum())
            for k in range(5):
                lp[k, j] += float(maps[k][m].sum())
        print(f"  {ci+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    tot = lp.sum()
    print(f"\n=== LPIPS budget of the PRODUCTION output by shipped-r band, HCM0181 n={len(stems)} ===")
    print(f"{'r band':>12} {'%pixels':>9} {'%LPIPS':>9} {'density':>9} | per-layer %of that layer")
    for j in range(NB):
        fp = 100 * npx[j] / npx.sum(); fl = 100 * lp[:, j].sum() / tot
        per = " ".join(f"{100*lp[k,j]/lp[k].sum():6.2f}" for k in range(5))
        print(f"{NAMES[j]:>12} {fp:9.3f} {fl:9.3f} {fl/max(fp,1e-9):9.3f} | {per}")
    print("  (layer order: " + " ".join(LAYERS) + ")")
    print(f"\nTOTAL LPIPS/img {tot/len(stems):.5f}")
    cum = 0.0
    for j in range(NB - 1, -1, -1):
        cum += 100 * lp[:, j].sum() / tot
        print(f"  r >= {BINS[j]:<6}: {100*npx[j:].sum()/npx.sum():6.2f}% of pixels "
              f"hold {cum:6.2f}% of LPIPS")


if __name__ == "__main__":
    main()

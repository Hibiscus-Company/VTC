"""P2: 1-D sweeps of image transforms, scored with ALL THREE metrics separately, so we can see
where the COMPOSITE optimum sits vs the MSE optimum.

Sweeps:
  shift  : global sub-pixel translation (bicubic), dx,dy in [-0.6,0.6]
  sharp  : unsharp mask  out = x + amt*(x - G_sigma*x)
  blur   : gaussian blur, sigma sweep (amt<0 side of sharp)
  gamma  : out = x**g  (sanity: known-dead axis, used as a calibration control)
"""
import os, argparse, json, itertools
import numpy as np
import torch
import torch.nn.functional as F
import mlib

torch.set_num_threads(int(os.environ.get("NT", "24")))


def gauss_k(sigma, device="cpu"):
    r = max(1, int(np.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g.to(device), r


def blur(x, sigma):
    if sigma <= 0:
        return x
    g, r = gauss_k(sigma, x.device)
    c = x.shape[1]
    kx = g.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = g.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), ky, groups=c)
    return y


def shift(x, dx, dy):
    """Positive dx shifts content to the RIGHT by dx px. Bicubic, replicate border."""
    if dx == 0 and dy == 0:
        return x
    B, C, H, W = x.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    gx = (xs - dx) / (W - 1) * 2 - 1
    gy = (ys - dy) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(x, grid, mode="bicubic", padding_mode="border", align_corners=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mode", required=True, choices=["shift", "sharp", "gamma", "shift1d"])
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()

    lp = mlib.LP("cpu")
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(a.gt_dir)}
    files = sorted(f for f in os.listdir(a.render_dir) if f.lower().endswith((".png", ".jpg")))
    files = [f for f in files if os.path.splitext(f)[0] not in a.skip]
    step = max(1, len(files) // a.n)
    files = files[::step][:a.n]

    if a.mode == "shift":
        grid = [(dx, dy) for dx in (-0.4, -0.2, 0.0, 0.2, 0.4) for dy in (-0.4, -0.2, 0.0, 0.2, 0.4)]
        names = [f"dx{dx:+.2f}_dy{dy:+.2f}" for dx, dy in grid]
        fn = lambda x, p: shift(x, p[0], p[1])
    elif a.mode == "shift1d":
        grid = [(dx, 0.0) for dx in (-0.6, -0.45, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.45, 0.6)]
        names = [f"dx{dx:+.2f}" for dx, _ in grid]
        fn = lambda x, p: shift(x, p[0], p[1])
    elif a.mode == "sharp":
        grid = [("blur", 0.4), ("blur", 0.25), ("id", 0.0),
                ("sh", 0.10), ("sh", 0.20), ("sh", 0.35), ("sh", 0.50), ("sh", 0.75), ("sh", 1.00),
                ("sh2", 0.20), ("sh2", 0.40), ("sh2", 0.70)]
        names = [f"{k}{v:.2f}" for k, v in grid]

        def fn(x, p):
            k, v = p
            if k == "id":
                return x
            if k == "blur":
                return blur(x, v)
            sig = 0.8 if k == "sh" else 1.6
            return (x + v * (x - blur(x, sig))).clamp(0, 1)
    else:
        grid = [0.96, 0.98, 0.99, 1.0, 1.01, 1.02, 1.04]
        names = [f"g{g:.2f}" for g in grid]
        fn = lambda x, p: x.clamp(1e-6, 1) ** p

    acc = {n: [0.0, 0.0, 0.0, 0] for n in names}
    for i, f in enumerate(files):
        s = os.path.splitext(f)[0]
        r = mlib.to_t(mlib.load_u8(os.path.join(a.render_dir, f)))
        g = mlib.to_t(mlib.load_u8(os.path.join(a.gt_dir, gt_by[s])))
        for n, p in zip(names, grid):
            y = fn(r, p).clamp(0, 1)
            acc[n][0] += mlib.psnr(y, g)
            acc[n][1] += float(mlib.ssim(y, g))
            acc[n][2] += lp(y, g)
            acc[n][3] += 1
        print(f"  [{i+1}/{len(files)}] {s}", flush=True)

    print(f"\n=== {a.tag} mode={a.mode} n={len(files)} ===")
    rows = []
    for n in names:
        P, S, L, k = acc[n]
        P, S, L = P / k, S / k, L / k
        rows.append((n, P, S, L, mlib.score(P, S, L)))
    ref = [r for r in rows if r[0] in ("dx+0.00_dy+0.00", "dx+0.00", "id0.00", "g1.00")][0]
    for n, P, S, L, sc in rows:
        print(f"{n:18s} PSNR {P:7.4f} ({P-ref[1]:+.4f})  SSIM {S:.5f} ({S-ref[2]:+.5f})  "
              f"LPIPS {L:.5f} ({L-ref[3]:+.5f})  SCORE {sc:8.4f} ({sc-ref[4]:+.4f})")
    json.dump(rows, open(f"/mnt/d/avv/metric_probe/p2_{a.tag}_{a.mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

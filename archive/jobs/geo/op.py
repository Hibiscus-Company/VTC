#!/usr/bin/env python
"""REAL OPERATORS aimed at flat-region LPIPS. All test-time computable (masks from our own
output, never from GT). Scored through the FULL shipped chain incl. the JPEG round-trip.

  base       shipped: 4-member mean -> energy restore lam=1.0 -> lens field lanczos4 -> q100/ss2
  ditherN    + uniform dither of N LSB in flat regions, applied before the 8-bit quantisation.
             Tests the BANDING hypothesis. Costs (N/255)^2/12 of variance = ~0 in PSNR/SSIM.
  jartQ      + JPEG-artifact texture injected into flat regions: out = X + a*m*(decode(enc(X,Q))-X).
             Follows the one measured fact that says structured incoherent texture in flat
             regions can PAY (shipped q100/ss2 beats lossless PNG).
  injflat    + member-0 texture injection restricted to flat regions, at the L0 band:
             L0 <- L0 + a*m*(L0_member0 - L0_mean). Global version measured +0.0147 at k=4.

CONTROL: `dither_edge` puts the identical dither in NON-flat regions. If the flat/edge arms move
together the effect is not geographic.
"""
import io, os, sys, time
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from energy_restore import restore
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
rng = np.random.default_rng(0)


def loadt(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def enc_bytes(x, **kw):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **kw)
    return b.getvalue()


def dec(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float32) / 255.0


def restore_multi(ens, members, lams, k, win=3, nlev=5, clamp=4.0):
    """Energy restore applied at EVERY level in `lams` (dict level->lam). lams={0:1.0} is the
    shipped operator exactly."""
    K = _K
    laps, res, sizes = lap_pyr(ens, nlev, K)
    mlaps = [lap_pyr(m, nlev, K)[0] for m in members]
    out = list(laps)
    for lvl, lam in lams.items():
        L = laps[lvl]
        Eb = boxf((L ** 2).sum(1, keepdim=True), win)
        V = 0.0
        for ml in mlaps:
            V = V + boxf(((ml[lvl] - L) ** 2).sum(1, keepdim=True), win)
        V = V / len(mlaps) * (k / (k - 1.0))
        r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
        out[lvl] = L * (1.0 + lam * (r - 1.0))
    return lap_recon(out, res, sizes, K)


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
stems = stems[::max(1, len(stems) // N)][:N]
cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")

dev = "cpu"   # both GPUs are running production training; this touches neither
torch.set_num_threads(10)
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

ARMS = ["base", "dither1", "dither_edge1", "jart85", "injflat", "L1r0.5", "L1r1.0"]
acc = {a: [0.0, 0.0, 0.0] for a in ARMS}
nby = {a: 0 for a in ARMS}
t0 = time.time()
for si, s in enumerate(stems):
    mem = [loadt(os.path.join(d, s + ".png")) for d in MEM]
    ens = torch.stack(mem).mean(0)
    out = restore(ens, mem, 1.0, len(mem), 3).clamp(0, 1)
    X = np.clip(warp(out[0].permute(1, 2, 0).numpy(), lens, "lanczos"), 0, 1)
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = X.shape

    xg = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(xg, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(xg, cv2.CV_32F, 0, 1, 3)
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3.0)
    t_lo = np.percentile(gm, 35.0)
    m = cv2.GaussianBlur((gm <= t_lo).astype(np.float32), (0, 0), 4.0)[..., None]

    # --- injflat: member-0 L0 texture, flat regions only, built BEFORE the warp
    K = _K
    laps, res, sizes = lap_pyr(ens, 5, K)
    L0m = lap_pyr(mem[0], 5, K)[0][0]
    rest = restore(ens, mem, 1.0, len(mem), 3)          # shipped operator output
    dl = (L0m - laps[0])
    mt = torch.from_numpy(np.ascontiguousarray(m.transpose(2, 0, 1)[None]))
    inj = rest + 1.0 * mt * dl
    Xinj = np.clip(warp(inj.clamp(0, 1)[0].permute(1, 2, 0).numpy(), lens, "lanczos"), 0, 1)

    # --- level-1 energy restoration ON TOP of the shipped level-0 restore
    if si == 0:
        chk = float((restore_multi(ens, mem, {0: 1.0}, len(mem)) - rest).abs().max())
        assert chk < 1e-6, f"restore_multi != shipped restore ({chk})"
        print(f"  [check] restore_multi({{0:1.0}}) == shipped restore, max|diff|={chk:.2e}",
              flush=True)

    def wr(t):
        return np.clip(warp(t.clamp(0, 1)[0].permute(1, 2, 0).numpy(), lens, "lanczos"), 0, 1)
    Xl1a = wr(restore_multi(ens, mem, {0: 1.0, 1: 0.5}, len(mem)))
    Xl1b = wr(restore_multi(ens, mem, {0: 1.0, 1: 1.0}, len(mem)))

    u = (rng.random((H, W, 1), dtype=np.float32) - 0.5) / 255.0
    outs = {
        "base": X,
        "dither1": X + m * u,
        "dither_edge1": X + (1.0 - m) * u,
        "jart85": X + m * (dec(enc_bytes(X, quality=85, subsampling=2)) - X),
        "injflat": Xinj,
        "L1r0.5": Xl1a,
        "L1r1.0": Xl1b,
    }
    gt_t = torch.from_numpy(np.ascontiguousarray(G)).permute(2, 0, 1).unsqueeze(0).to(dev)
    for a in ARMS:
        bts = enc_bytes(outs[a], **SHIPPED_JPEG)
        nby[a] += len(bts)
        r = torch.from_numpy(np.ascontiguousarray(dec(bts))).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            acc[a][0] += 10 * np.log10(1.0 / max(((r - gt_t) ** 2).mean().item(), 1e-12))
            acc[a][1] += float(repo_ssim(r, gt_t))
            acc[a][2] += float(vgg(r * 2 - 1, gt_t * 2 - 1).item())
        del r
    del gt_t
    k = si + 1
    line = f"  {k}/{len(stems)} {time.time()-t0:.0f}s |"
    b0 = None
    for a in ARMS:
        P, S, L = (v / k for v in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if a == "base":
            b0 = sc; line += f" base={sc:.4f}"
        else:
            line += f" {a}={sc-b0:+.4f}"
    print(line, flush=True)

n = len(stems)
print(f"\n=== FLAT-REGION OPERATORS, HCM0181, n={n}, full shipped chain ===")
print(f"{'arm':>13} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs base':>9} {'MB/60':>8}")
base = None
for a in ARMS:
    P, S, L = (x / n for x in acc[a])
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    if a == "base":
        base = sc
    print(f"{a:>13} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {sc-base:+9.4f} "
          f"{nby[a]/1e6*60/n:8.2f}")

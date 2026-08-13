#!/usr/bin/env python
"""THE OTHER HALF OF THE LPIPS BUDGET: flat regions and sky.

Everything we ship acts on textured/edge content. Energy restoration boosts the finest band where
members DISAGREE, which is exactly where texture is. But the error decomposition says our LPIPS
does not live there:
    region                   %px    %squared error   %LPIPS
    near strong edge (<=3px) 27.1       67.9          24.1
    far from edge  (>12px)   34.7       10.0        **39.0**
    sky                       2.6        0.5     3.6 (density 1.377, highest of any region)
LPIPS's per-location feature normalisation makes SMALL ABSOLUTE ERRORS IN SMOOTH AREAS expensive --
relu2_2, which carries 25.4% of LPIPS, has flat-region density 1.259 against 0.607 at strong edges.
So 39% of the LPIPS budget sits in regions no operator of ours touches.

HYPOTHESIS: in flat regions the mean's finest band is mostly 3DGS noise, not signal -- GT sky really
is smooth. Two independent results already point this way: the round-trip study found an INTERIOR
optimum in warp sharpness (lanczos, which attenuates 2.4% of HF, beats a near-ideal warp that
attenuates 0.5%), and the power-spectrum study found 96% of our missing HF is INCOHERENT with GT.

OPERATOR -- the natural completion of energy restoration, same band, opposite sign:
    out = mean + [ lam*(r-1) - mu*f ] * L0(mean)
    r = sqrt(1 + (k/(k-1)) * E(L0_i - L0_mean)/E(L0_mean))     boost where members disagree
    f = 1 / (1 + E(L0_mean)/tau)                               attenuate where the mean is FLAT
tau = per-image median of E(L0_mean), so f -> 1 in smooth regions and -> 0 wherever there is real
fine detail. mu=0 reproduces r28 exactly, which is the control.

This is cleanup-class, the class that has inverted on us three times (EMA, the JPEG encode, the
restoration head) -- which is precisely why it is measured on the PRODUCTION harness (real test
poses, real test GT, models trained on 100% of train photos), the regime where that bias does not
exist. r27 transferred from this harness to the leaderboard at 96% with submetric accuracy.
"""
import io, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
MEM = [f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
       for m in ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]


def precompute(ens, members, win=3, nlev=5):
    """Everything that does not depend on (lam, mu). The first version rebuilt all member
    pyramids inside every arm -- 6x redundant, and on a machine already running two trainings
    that redundancy starved the GPUs of CPU and stalled them to 0% util between steps."""
    K = _K.to(ens.device)
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for m in members:
        V = V + boxf(((lap_pyr(m, nlev, K)[0][0] - L0) ** 2).sum(1, keepdim=True), win)
    k = float(len(members))                # the ensemble IS the mean of exactly these members
    V = V / len(members) * (k / (k - 1.0))  # unbiased: deviations are about the mean they formed
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0)
    tau = torch.quantile(Eb.flatten().float(), 0.5).clamp_min(1e-10)
    f = 1.0 / (1.0 + Eb / tau)
    return dict(K=K, laps=laps, res=res, sizes=sizes, L0=L0, r=r, f=f)


def operator(pc, lam, mu):
    g = (1.0 + lam * (pc["r"] - 1.0) - mu * pc["f"]).clamp_min(0.0)
    return lap_recon([pc["L0"] * g] + pc["laps"][1:], pc["res"], pc["sizes"], pc["K"])


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic")

    ARMS = [(1.0, 0.0), (1.0, 0.10), (1.0, 0.20), (1.0, 0.35), (1.0, 0.50), (0.0, 0.20)]
    names = [f"lam{l}_mu{m}" for l, m in ARMS]
    acc = {n: [0.0, 0.0, 0.0] for n in names}
    nb = {n: 0 for n in names}
    t0 = time.time()
    for c, s in enumerate(stems):
        mem = [torch.from_numpy(np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                                           dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
               for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        pc = precompute(ens, mem)
        for (lam, mu), n in zip(ARMS, names):
            o = operator(pc, lam, mu).clamp(0, 1)[0].permute(1, 2, 0).numpy()
            x = np.clip(warp(o, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            nb[n] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            rr = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[n][0] += 10 * np.log10(1.0 / max(((rr - g) ** 2).mean().item(), 1e-12))
                acc[n][1] += float(repo_ssim(rr, g))
                acc[n][2] += float(vgg(rr * 2 - 1, g * 2 - 1).item())
        if c % 10 == 0:
            print(f"  {c}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    n_ = len(stems)
    print(f"\nFLAT-BAND ATTENUATION on top of energy restoration, {TAG}, n={n_}, full shipped chain")
    print(f"{'arm':>16} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'vs mu=0':>9} {'MB':>7}")
    base = None
    for nm in names:
        P, S, L = (x / n_ for x in acc[nm])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if nm == "lam1.0_mu0.0":
            base = sc
        print(f"{nm:>16} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {sc-base:+9.4f} {nb[nm]/1e6:7.2f}")


if __name__ == "__main__":
    main()

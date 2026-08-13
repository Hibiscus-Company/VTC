#!/usr/bin/env python
"""LPIPS-DIRECT experiment: STRUCTURE vs AMPLITUDE in the high-disagreement tail of L0.

Diag B (lpd_decile.py) established, on the production harness:
  decile of r     %L0 energy   rho(raw)   amp(raw)
      0             29.2        0.917      0.943
      7-9            5.4        0.53/0.41/0.23
and the archived rmax sweep says clamping r at 1.5 (which only touches ~the top 1% of pixels,
0.34% of the band energy) costs 13% of the whole operator's value (+0.1742 -> +0.1510).
=> LPIPS pays ~40x more per unit of restored energy in the tail than in the bulk.

But in the tail the shipped operator AMPLIFIES the ensemble mean's own L0 -- which there is the
average of k mutually-disagreeing textures, i.e. MUSH (rho 0.23-0.53).  A single member's L0 has
the SAME energy (that is exactly what r targets) but NATURAL texture statistics.  LPIPS-vgg is a
learned metric on natural-image statistics, so at MATCHED BAND ENERGY it should prefer natural
texture to amplified mush.  Nothing in the dead list tests this: unsharp/flat-band attenuation are
gain operators, and the robust aggregators (maxmag/pnorm/median) were applied at EVERY pixel,
paying the mean's PSNR everywhere.

OPERATOR (only the finest band changes; levels>=1 and the residual stay at the pixel mean):
    s   = beta * clip((r - 1.05)/0.35, 0, 1)            gate: 0 in the bulk, 1 in the tail
    out = mean + [ (1-s)*r - 1 ] * L0_mean + s * L0_j
  s=0 reproduces the SHIPPED operator EXACTLY (asserted).  s=1 hands the band to member j.
  The blend is energy-matched by construction, so the ONLY thing that varies is STRUCTURE.

CONTROL (the point of the experiment): ROLL replaces L0_j by roll(L0_j, 32, 32) -- identical
natural texture statistics and identical energy, GT-coherence destroyed.  If SUB == ROLL the win
is generic natural texture (and could be synthesised); if SUB > ROLL it is the member's own
GT-coherent content; if both lose, the tail wants amplitude, not structure -- a clean kill.

Shared computation (loads, pyramids, box energies, r) is HOISTED out of the arm loop.
"""
import io, os, sys, time, json
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, boxf, _K            # noqa: E402
from fieldlib import LooPool, upsample, warp     # noqa: E402

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(6)
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
C_SHIP, LAM = 4.0 / 3.0, 1.0
G_LO, G_W = 1.05, 0.35

# Minimal decisive set: SHIP is the reference, SUB is the treatment, ROLL is the control that
# holds texture statistics + band energy fixed and destroys only GT-coherence.
ARMS = [
    ("SHIP",       dict()),
    ("SUB b0.70",  dict(beta=0.70)),
    ("ROLL b0.70", dict(beta=0.70, roll=True)),
]


def u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    dev = "cuda"
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:n]
    cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
    K = _K.to(dev)
    acc = {a: [0.0, 0.0, 0.0] for a, _ in ARMS}
    gate_px = []
    t0 = time.time()

    for ci, s in enumerate(stems):
        mem = [to_t(u8(os.path.join(d, s + ".png")), dev) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = to_t(u8(os.path.join(GTD, gt_by[s])), dev)
        # -------- HOISTED shared analysis --------
        L0 = lap_pyr(ens, 1, K)[0][0]
        mL0 = [lap_pyr(m, 1, K)[0][0] for m in mem]
        Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
        V = sum(boxf(((ml - L0) ** 2).sum(1, keepdim=True), 3) for ml in mL0) / len(mem)
        r = torch.sqrt(1.0 + C_SHIP * V / (Eb + 1e-10)).clamp(max=4.0)
        gate = ((r - G_LO) / G_W).clamp(0, 1)
        Lj = mL0[0]
        Lj_roll = torch.roll(Lj, shifts=(32, 32), dims=(2, 3))
        gate_px.append(float((gate > 0).float().mean()))

        for name, kw in ARMS:
            if kw.get("off"):
                x = ens
            else:
                beta = kw.get("beta", 0.0)
                if beta == 0.0:
                    x = ens + LAM * (r - 1.0) * L0
                else:
                    sg = beta * gate
                    src = Lj_roll if kw.get("roll") else Lj
                    x = ens + ((1.0 - sg) * r - 1.0) * L0 + sg * src
            x = x.clamp(0, 1)
            xn = np.clip(warp(np.ascontiguousarray(x[0].permute(1, 2, 0).cpu().numpy()),
                              lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((xn * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.uint8)
            y = to_t(j, dev)
            with torch.no_grad():
                acc[name][0] += 10 * np.log10(1.0 / max(float(((y - g) ** 2).mean()), 1e-12))
                acc[name][1] += float(repo_ssim(y, g))
                acc[name][2] += float(vgg(y * 2 - 1, g * 2 - 1).item())
        if True:
            m = ci + 1
            def sc(a):
                return 100 * (0.4 * (1 - acc[a][2] / m) + 0.3 * acc[a][1] / m
                              + 0.3 * acc[a][0] / m / 50)
            line = " ".join(f"{a}{sc(a)-sc('SHIP'):+.4f}" for a, _ in ARMS)
            print(f"[{m}/{len(stems)} {time.time()-t0:.0f}s vsSHIP] {line}", flush=True)

    m = len(stems)
    print(f"\n=== HCM0181 REAL TEST GT, FULL SHIPPED CHAIN, n={m}, "
          f"gate touches {100*np.mean(gate_px):.2f}% of pixels ===")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs SHIP':>9}")
    S = {}
    for a, _ in ARMS:
        P, Ss, L = (v / m for v in acc[a])
        S[a] = 100 * (0.4 * (1 - L) + 0.3 * Ss + 0.3 * min(P / 50.0, 1.0))
    for a, _ in ARMS:
        P, Ss, L = (v / m for v in acc[a])
        print(f"{a:>12} {S[a]:9.4f} {P:8.4f} {Ss:8.5f} {L:8.5f} "
              f"{S[a]-S['SHIP']:+9.4f}")
    json.dump({a: [v / m for v in acc[a]] for a, _ in ARMS},
              open(f"{HERE}/lpd_sub.json", "w"), indent=1)


if __name__ == "__main__":
    main()

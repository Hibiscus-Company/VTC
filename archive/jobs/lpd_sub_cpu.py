#!/usr/bin/env python
"""Same experiment as lpd_sub.py, scored ENTIRELY ON CPU.

Both GPUs are running production member training AND ~8 other agents' scoring jobs; nvidia-smi
itself blocks, and the CUDA build could not finish a single image in 17 minutes.  LPIPS-vgg and
repo_ssim are deterministic across devices to ~1e-6, which is 3 orders below the deltas at stake,
so the CPU path yields the same numbers while touching no GPU at all.

Cost reduction: the GT VGG features are computed ONCE per image and reused by every arm
(hoisting, as required) -- lpips.LPIPS.forward would recompute them per call.

  s   = beta * clip((r - 1.05)/0.35, 0, 1)
  out = mean + [ (1-s)*r - 1 ] * L0_mean + s * L0_j
  SHIP = s==0 exactly.  ROLL = same but L0_j replaced by roll(L0_j,32,32): identical natural
  texture statistics and identical band energy, GT-coherence destroyed.
"""
import io, os, sys, time, json
import numpy as np
import torch
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, boxf, _K            # noqa: E402
from fieldlib import LooPool, upsample, warp     # noqa: E402
import lpips as L                                 # noqa: E402

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(10)
cv2.setNumThreads(4)
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
C_SHIP, LAM = 4.0 / 3.0, 1.0
G_LO, G_W = 1.05, 0.35

ARMS = [
    ("SHIP",       dict()),
    ("SUB b0.70",  dict(beta=0.70)),
    ("ROLL b0.70", dict(beta=0.70, roll=True)),
]


def u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    from utils.loss_utils import ssim as repo_ssim
    m_ = L.LPIPS(net="vgg", verbose=False).eval()

    @torch.no_grad()
    def feats(t):
        return [L.normalize_tensor(f) for f in m_.net.forward(m_.scaling_layer(t * 2 - 1))]

    @torch.no_grad()
    def lp(fa, fb):
        return float(sum(m_.lins[k]((fa[k] - fb[k]) ** 2).mean() for k in range(m_.L)))

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:n]
    cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
    K = _K
    acc = {a: [0.0, 0.0, 0.0] for a, _ in ARMS}
    per = {a: [] for a, _ in ARMS}
    gate_px = []
    t0 = time.time()

    for ci, s in enumerate(stems):
        mem = [to_t(u8(os.path.join(d, s + ".png"))) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = to_t(u8(os.path.join(GTD, gt_by[s])))
        fg = feats(g)                                    # HOISTED: GT features once per image
        L0 = lap_pyr(ens, 1, K)[0][0]
        mL0 = [lap_pyr(m, 1, K)[0][0] for m in mem]
        Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
        V = sum(boxf(((ml - L0) ** 2).sum(1, keepdim=True), 3) for ml in mL0) / len(mem)
        r = torch.sqrt(1.0 + C_SHIP * V / (Eb + 1e-10)).clamp(max=4.0)
        gate = ((r - G_LO) / G_W).clamp(0, 1)
        Lj, Ljr = mL0[0], torch.roll(mL0[0], shifts=(32, 32), dims=(2, 3))
        gate_px.append(float((gate > 0).float().mean()))

        for name, kw in ARMS:
            beta = kw.get("beta", 0.0)
            if beta == 0.0:
                x = ens + LAM * (r - 1.0) * L0
            else:
                sg = beta * gate
                x = ens + ((1.0 - sg) * r - 1.0) * L0 + sg * (Ljr if kw.get("roll") else Lj)
            x = x.clamp(0, 1)
            xn = np.clip(warp(np.ascontiguousarray(x[0].permute(1, 2, 0).numpy()), lens,
                              "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((xn * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            y = to_t(np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                                dtype=np.uint8))
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(float(((y - g) ** 2).mean()), 1e-12))
                Sv = float(repo_ssim(y, g))
                Lv = lp(feats(y), fg)
            acc[name][0] += P; acc[name][1] += Sv; acc[name][2] += Lv
            per[name].append(100 * (0.4 * (1 - Lv) + 0.3 * Sv + 0.3 * P / 50))
        m = ci + 1
        def sc(a):
            return 100 * (0.4 * (1 - acc[a][2] / m) + 0.3 * acc[a][1] / m
                          + 0.3 * acc[a][0] / m / 50)
        print(f"[{m}/{len(stems)} {time.time()-t0:.0f}s] SHIP {sc('SHIP'):.4f} | " +
              " ".join(f"{a}{sc(a)-sc('SHIP'):+.4f}" for a, _ in ARMS[1:]), flush=True)

    m = len(stems)
    print(f"\n=== HCM0181 REAL TEST GT, FULL SHIPPED CHAIN (restore->median field lanczos4->"
          f"JPEG q100/ss2), n={m}, gate touches {100*np.mean(gate_px):.2f}% of pixels ===")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs SHIP':>9} "
          f"{'win/n':>7}")
    S = {a: 100 * (0.4 * (1 - acc[a][2] / m) + 0.3 * acc[a][1] / m + 0.3 * acc[a][0] / m / 50)
         for a, _ in ARMS}
    for a, _ in ARMS:
        P, Sv, Lv = (v / m for v in acc[a])
        w = sum(1 for i in range(m) if per[a][i] > per["SHIP"][i])
        print(f"{a:>12} {S[a]:9.4f} {P:8.4f} {Sv:8.5f} {Lv:8.5f} {S[a]-S['SHIP']:+9.4f} "
              f"{w:4d}/{m}")
    json.dump({"acc": {a: [v / m for v in acc[a]] for a, _ in ARMS}, "per": per, "n": m},
              open(f"{HERE}/lpd_sub_cpu.json", "w"), indent=1)


if __name__ == "__main__":
    main()

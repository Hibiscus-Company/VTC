#!/usr/bin/env python
"""FAULT AUDIT of the r28 energy-restoration operator, on the production harness.

All shared computation (member load, Laplacian pyramids, box energies) is hoisted OUT of the
arm loop -- one pyramid set per image, reused by every arm.  GPU is touched only for LPIPS/SSIM.
"""
import argparse, io, os, sys, time, json
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K   # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"


def u8_load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def quant(t):
    """float [0,1] tensor -> uint8 round-trip, exactly as production writes/reads a PNG."""
    return torch.round(t.clamp(0, 1) * 255.0) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=f"{HERE}/audit_er.json")
    args = ap.parse_args()
    dev = "cuda"
    from utils.loss_utils import ssim as repo_ssim
    from fieldlib import LooPool, upsample, warp
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:args.n]
    cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
    K = _K.to(dev)
    NLEV, C_SHIP, LAM = 5, 4.0 / 3.0, 1.0     # harness k=4 -> k/(k-1)=4/3

    # ---------------- arms: (name, kwargs) ----------------
    ARMS = [
        ("base",        dict(off=True)),
        ("SHIP",        dict()),
        ("prod_u8",     dict(u8=True)),                       # production's 2 extra PNG rounds
        ("c=1.000",     dict(c=1.0)),                         # no Bessel factor
        ("c=1.143",     dict(c=8.0 / 7.0)),                   # what r29 towers ship (k=8)
        ("c=1.167",     dict(c=7.0 / 6.0)),                   # what r28 towers ship (k=7)
        ("c=1.600",     dict(c=1.6)),
        ("clamp=1e9",   dict(clamp=1e9)),
        ("win=5",       dict(win=5)),
        ("L1too",       dict(l1=1.0)),                        # level-1 restoration
        ("perchan",     dict(perchan=True)),                  # per-channel r instead of RGB-summed
    ]
    acc = {a: [0.0, 0.0, 0.0] for a, _ in ARMS}
    diag = dict(recon_err=[], rmean=[], rp999=[], clamp_hit=[], oob=[], wratio=[],
                E0_ratio=[], E1_ratio=[])

    t0 = time.time()
    for ci, s in enumerate(stems):
        mem_u8 = [u8_load(os.path.join(d, s + ".png")) for d in MEM]
        mem = [to_t(a, dev) for a in mem_u8]
        ens = torch.stack(mem).mean(0)
        g = to_t(u8_load(os.path.join(GTD, gt_by[s])), dev)

        # ---- HOISTED shared analysis (float chain) ----
        laps, res, sizes = lap_pyr(ens, NLEV, K)
        L0 = laps[0]
        mlaps = [lap_pyr(m, NLEV, K)[0] for m in mem]
        pre = {}
        for w in (1, 3, 5):
            Eb = boxf((L0 ** 2).sum(1, keepdim=True), w)
            V = sum(boxf(((ml[0] - L0) ** 2).sum(1, keepdim=True), w) for ml in mlaps) / len(mem)
            pre[w] = (Eb, V)
        Eb3, V3 = pre[3]
        Ebc = boxf(L0 ** 2, 3)                                   # per-channel energy
        Vc = sum(boxf((ml[0] - L0) ** 2, 3) for ml in mlaps) / len(mem)
        # level 1
        L1 = laps[1]
        Eb1 = boxf((L1 ** 2).sum(1, keepdim=True), 3)
        V1 = sum(boxf(((ml[1] - L1) ** 2).sum(1, keepdim=True), 3) for ml in mlaps) / len(mem)

        # ---- HOISTED: production uint8 chain analysis ----
        ens_q = quant(ens)
        lapsq, resq, sizesq = lap_pyr(ens_q, NLEV, K)
        L0q = lapsq[0]
        Ebq = boxf((L0q ** 2).sum(1, keepdim=True), 3)
        Vq = sum(boxf(((ml[0] - L0q) ** 2).sum(1, keepdim=True), 3) for ml in mlaps) / len(mem)

        # ---- HOISTED: padded (boundary) analysis ----
        P = 64
        ensp = torch.nn.functional.pad(ens, (P, P, P, P), mode="reflect")
        lapsp, resp, sizesp = lap_pyr(ensp, NLEV, K)
        L0p = lapsp[0]
        Ebp = boxf((L0p ** 2).sum(1, keepdim=True), 3)
        Vp = 0.0
        for m in mem:
            mp0 = lap_pyr(torch.nn.functional.pad(m, (P, P, P, P), mode="reflect"), NLEV, K)[0][0]
            Vp = Vp + boxf(((mp0 - L0p) ** 2).sum(1, keepdim=True), 3)
        Vp = Vp / len(mem)

        # ---- diagnostics ----
        if ci < 8:
            rec = lap_recon(laps, res, sizes, K)
            diag["recon_err"].append(float((rec - ens).abs().max()))
            r = torch.sqrt(1.0 + C_SHIP * V3 / (Eb3 + 1e-10))
            diag["rmean"].append(float(r.mean()))
            diag["rp999"].append(float(torch.quantile(r.flatten().float()[::7], 0.999)))
            diag["clamp_hit"].append(float((r > 4.0).float().mean()))
            out = ens + LAM * (r.clamp(max=4.0) - 1.0) * L0
            diag["oob"].append(float(((out < 0) | (out > 1)).float().mean()))
            # energy ratio: is level>=1 really "already at GT energy" at this k?
            glaps = lap_pyr(g, NLEV, K)[0]
            diag["E0_ratio"].append(float((L0 ** 2).mean() / (glaps[0] ** 2).mean()))
            diag["E1_ratio"].append(float((L1 ** 2).mean() / (glaps[1] ** 2).mean()))
            # weighted-mean check: does k/(k-1)*unweighted-dev-mean equal the correct target?
            wv = torch.tensor([0.4, 0.2, 0.2, 0.2], device=dev)
            mw = sum(float(wv[i]) * mem[i] for i in range(4))
            L0w = lap_pyr(mw, NLEV, K)[0][0]
            dev_u = sum(boxf(((ml[0] - L0w) ** 2).sum(1, keepdim=True), 3) for ml in mlaps) / 4
            dev_w = sum(float(wv[i]) * boxf(((mlaps[i][0] - L0w) ** 2).sum(1, keepdim=True), 3)
                        for i in range(4))
            keff = 1.0 / float((wv ** 2).sum())
            code = dev_u * (4.0 / 3.0)                     # what the code does with --k 4
            corr = dev_w * (keff / (keff - 1.0))           # Kish-correct
            diag["wratio"].append(float((code.sum() / corr.sum())))

        # ---------------- arms ----------------
        for name, kw in ARMS:
            if kw.get("off"):
                x = ens
            elif kw.get("u8"):
                c = kw.get("c", C_SHIP)
                r = torch.sqrt(1.0 + c * Vq / (Ebq + 1e-10)).clamp(max=4.0)
                x = quant(ens_q + LAM * (r - 1.0) * L0q)
            elif kw.get("pad"):
                c = kw.get("c", C_SHIP)
                r = torch.sqrt(1.0 + c * Vp / (Ebp + 1e-10)).clamp(max=4.0)
                xp = ensp + LAM * (r - 1.0) * L0p
                x = xp[:, :, P:-P, P:-P]
            elif kw.get("perchan"):
                r = torch.sqrt(1.0 + C_SHIP * Vc / (Ebc + 1e-10)).clamp(max=4.0)
                x = ens + LAM * (r - 1.0) * L0
            else:
                c = kw.get("c", C_SHIP)
                w = kw.get("win", 3)
                Eb, V = pre[w]
                r = torch.sqrt(1.0 + c * V / (Eb + 1e-10)).clamp(max=kw.get("clamp", 4.0))
                if kw.get("l1"):
                    r1 = torch.sqrt(1.0 + c * V1 / (Eb1 + 1e-10)).clamp(max=4.0)
                    x = lap_recon([L0 * (1.0 + LAM * (r - 1.0)),
                                   L1 * (1.0 + kw["l1"] * (r1 - 1.0))] + laps[2:],
                                  res, sizes, K)
                else:
                    x = ens + LAM * (r - 1.0) * L0
            if not kw.get("noclip"):
                x = x.clamp(0, 1)
            xn = x[0].permute(1, 2, 0).cpu().numpy()
            xn = np.clip(warp(np.ascontiguousarray(xn), lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((xn * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.uint8)
            y = to_t(j, dev)
            with torch.no_grad():
                acc[name][0] += 10 * np.log10(1.0 / max(float(((y - g) ** 2).mean()), 1e-12))
                acc[name][1] += float(repo_ssim(y, g))
                acc[name][2] += float(vgg(y * 2 - 1, g * 2 - 1).item())
        if (ci + 1) % 5 == 0:
            n = ci + 1
            bs = 100 * (0.4 * (1 - acc["base"][2] / n) + 0.3 * acc["base"][1] / n
                        + 0.3 * acc["base"][0] / n / 50)
            line = " ".join(
                f"{a}{100*(0.4*(1-acc[a][2]/n)+0.3*acc[a][1]/n+0.3*acc[a][0]/n/50)-bs:+.4f}"
                for a, _ in ARMS[1:])
            print(f"[{n}/{len(stems)} {time.time()-t0:.0f}s] {line}", flush=True)

    n = len(stems)
    print("\n=== HCM0181 REAL TEST GT, FULL SHIPPED CHAIN "
          "(mean -> restore -> median field lanczos4 -> JPEG q100/ss2), n=%d ===" % n)
    print(f"{'arm':>10} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs base':>9} {'vs SHIP':>9}")
    sc = {}
    for a, _ in ARMS:
        P_, S_, L_ = (v / n for v in acc[a])
        sc[a] = 100 * (0.4 * (1 - L_) + 0.3 * S_ + 0.3 * min(P_ / 50.0, 1.0))
    for a, _ in ARMS:
        P_, S_, L_ = (v / n for v in acc[a])
        print(f"{a:>10} {sc[a]:9.4f} {P_:8.4f} {S_:8.5f} {L_:8.5f} "
              f"{sc[a]-sc['base']:+9.4f} {sc[a]-sc['SHIP']:+9.4f}")
    print("\nDIAG (first 8 images):")
    for k in diag:
        v = diag[k]
        if v:
            print(f"  {k:>10}: mean {np.mean(v):.6g}   min {np.min(v):.6g}  max {np.max(v):.6g}")
    json.dump({"n": n, "score": sc,
               "raw": {a: [v / n for v in acc[a]] for a, _ in ARMS},
               "diag": {k: list(map(float, v)) for k, v in diag.items()}},
              open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

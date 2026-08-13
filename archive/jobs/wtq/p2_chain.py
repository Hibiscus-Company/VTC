#!/usr/bin/env python
"""PASS 2: score arbitrary member WEIGHTINGS through the FULL shipped chain.

chain = weighted member mean -> energy restore (lam, weighted generalisation) -> median lens field
        warped with INTER_LANCZOS4 -> JPEG q100/ss2/optimize/progressive -> decode -> metrics
surface = production harness HCM0181, real test-pose GT.

SHARED WORK IS HOISTED: the level-0 Laplacian L0_i and its local energy A_i = box(sum_c L0_i^2) are
computed ONCE per member per image and reused by every arm.  Two identities make that exact:
  (1) the Laplacian pyramid is linear, so L0(sum w_i x_i) = sum w_i L0_i, and because the pyramid
      reconstructs exactly, restore(x) = x + lam*(r-1)*L0 -- no recon needed;
  (2) with sum w = 1,  sum_i w_i |L0_i - L0|^2 = sum_i w_i |L0_i|^2 - |L0|^2, and box() is linear,
      so the disagreement map costs ONE box filter per arm instead of k.
The weighted unbiased correction generalises k/(k-1) to 1/(1 - sum w_i^2)  (equal for w_i = 1/k).
A startup assert checks the fast path against the shipped energy_restore.restore() bit-for-bit.
"""
import os, sys, io, json, time, argparse
import numpy as np
import torch
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, boxf, _K
from fieldlib import LooPool, upsample

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(3)
torch.set_num_threads(3)
JPG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
OUT = "/mnt/d/avv/output/HCM0181_%s/test_poses_renders_png"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_sec", type=float, default=0.0)
    ap.add_argument("--parity", default="all", choices=("all","odd","even"))
    args = ap.parse_args()
    spec = json.load(open(args.arms))
    MEM = spec["members"]
    tag = spec.get("tag", TAG)
    gtd = spec.get("gtd", GTD)
    outpat = spec.get("dirpat", OUT)
    cachef = spec.get("cache", f"{HERE}/lens/cache/pub_{TAG}.npz")
    arms = spec["arms"]                       # [{name, w:[...]}]
    K = _K.to(DEV)

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(outpat % m, s + ".png")) for m in MEM))
    if args.parity in ("odd", "even"):        # LS arms are FIT on even images -> score on odd only
        want = 1 if args.parity == "odd" else 0
        stems = [s for i, s in enumerate(stems) if i % 2 == want]
    stems = stems[:args.n] if args.n > 0 else stems
    z = np.load(cachef)
    H, W = [int(x) for x in z["HW"]]
    lens = upsample(LooPool(z["s8"]).pooled("median"), H, W, "cubic")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MX = (xx + lens[..., 0]).astype(np.float32)
    MY = (yy + lens[..., 1]).astype(np.float32)

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(DEV).eval()

    per = {a["name"]: [] for a in arms}
    nby = {a["name"]: 0 for a in arms}
    t0 = time.time()
    checked = False
    for c, s in enumerate(stems):
        mem = [load(os.path.join(outpat % m, s + ".png")) for m in MEM]
        L0 = [lap_pyr(m, 5, K)[0][0] for m in mem]                      # HOISTED
        A = [boxf((l ** 2).sum(1, keepdim=True), 3) for l in L0]        # HOISTED
        g = load(os.path.join(gtd, gt_by[s]))
        if not checked:
            from energy_restore import restore as shipped
            w = np.full(len(mem), 1.0 / len(mem))
            ref = shipped(torch.stack([m[0] for m in mem]).mean(0).unsqueeze(0), mem,
                          args.lam, len(mem))
            fast = fast_restore(mem, L0, A, w, args.lam)
            d = float((ref - fast).abs().max())
            dm = float((ref - fast).abs().mean())
            print(f"[self-check] fast vs shipped restore  max|diff| = {d:.3e}  "
                  f"mean|diff| = {dm:.3e}", flush=True)
            assert dm < 1e-6, "fast path does not reproduce the shipped operator"
            checked = True
        for a in arms:
            w = np.asarray(a["w"], dtype=np.float64)
            o = fast_restore(mem, L0, A, w, args.lam, a.get("corr")).clamp(0, 1)
            x = o[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(cv2.remap(np.ascontiguousarray(x), MX, MY, cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(b, "JPEG", **JPG)
            nby[a["name"]] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(float(((r - g) ** 2).mean()), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            per[a["name"]].append((P, S, L, 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1))))
        del mem, L0, A, g
        if c % 2 == 1 or c == 0:
            print(f"  {c+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            dump(per, nby, arms, args)
        if args.max_sec and time.time() - t0 > args.max_sec:
            print(f"  wall budget reached at {c+1} images", flush=True)
            stems = stems[:c + 1]
            break

    n = len(stems)
    Aacc = {k: np.array(v) for k, v in per.items()}
    base = Aacc[arms[0]["name"]][:, 3]
    print(f"\n{tag} FULL SHIPPED CHAIN (weighted mean -> restore lam={args.lam} -> field lanczos4 "
          f"-> JPEG q100/ss2), n={n}")
    print(f"{'arm':>22} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs arm0':>9} "
          f"{'se':>7} {'wins':>6} {'MB':>7}")
    res = {}
    for a in arms:
        nm = a["name"]
        m = Aacc[nm].mean(0)
        d = Aacc[nm][:, 3] - base
        se = d.std(ddof=1) / np.sqrt(n) if nm != arms[0]["name"] else 0.0
        res[nm] = dict(score=m[3], psnr=m[0], ssim=m[1], lpips=m[2], d=float(d.mean()),
                       se=float(se), mb=nby[nm] / 1e6, wins=int((d > 0).sum()), w=list(a["w"]))
        print(f"{nm:>22} {m[3]:9.4f} {m[0]:8.4f} {m[1]:8.5f} {m[2]:8.5f} {d.mean():+9.4f} "
              f"{se:7.4f} {int((d>0).sum()):4d}/{n} {nby[nm]/1e6:7.2f}")
    json.dump(dict(res=res, n=n, members=MEM, lam=args.lam), open(args.out, "w"), indent=1)


def dump(per, nby, arms, args):
    A = {k: np.array(v) for k, v in per.items() if len(v)}
    if not A:
        return
    n = len(A[arms[0]["name"]])
    base = A[arms[0]["name"]][:, 3]
    res = {}
    for a in arms:
        nm = a["name"]
        m = A[nm].mean(0)
        d = A[nm][:, 3] - base
        res[nm] = dict(score=float(m[3]), psnr=float(m[0]), ssim=float(m[1]), lpips=float(m[2]),
                       d=float(d.mean()),
                       se=float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                       mb=nby[nm] / 1e6, wins=int((d > 0).sum()), w=list(a["w"]))
    json.dump(dict(res=res, n=n, lam=args.lam), open(args.out, "w"), indent=1)


def fast_restore(mem, L0s, As, w, lam, corr=None):
    """weighted mean + weighted energy restoration, using hoisted per-member L0 and box energies.
    corr overrides the unbiased correction 1/(1-sum w^2) -- used by the frozen-correction control
    that separates the weighting itself from the weighting's effect on the restore gain."""
    ens = sum(float(wi) * m for wi, m in zip(w, mem) if wi)
    l0 = sum(float(wi) * l for wi, l in zip(w, L0s) if wi)
    Eb = boxf((l0 ** 2).sum(1, keepdim=True), 3)
    Aw = sum(float(wi) * a for wi, a in zip(w, As) if wi)
    s2 = float((w ** 2).sum())
    if corr is None:
        if s2 >= 1.0 - 1e-9:                  # degenerate single-member weighting: no disagreement
            return ens
        corr = 1.0 / (1.0 - s2)
    V = (Aw - Eb).clamp_min(0.0) * float(corr)
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0)
    return ens + lam * (r - 1.0) * l0


if __name__ == "__main__":
    main()

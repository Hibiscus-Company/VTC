#!/usr/bin/env python
"""FIT-POOL DEPTH, WITH THE SUBSET-LUCK CONTROL target2.py was missing.

target2.py found B11's field at 120 fit views beats the same member's field at 60 views by
+0.0305 (t=11.0, 56/60) at matched amplitude.  But its "60" was NOT a random draw -- it was the
60 views that happen to intersect gsplatB9ut's render set.  A 1/n noise model predicts the
30->60 step should be TWICE the 60->120 step; the measurement says it is SMALLER (+0.0349 vs
+0.0305), which is the signature of a confounded subset, not of estimator noise.  So: three
INDEPENDENT random 60-draws from the same 120-view stack, plus 30 and 90, plus the fixed 60.

This matters for production: the shipped private fields are fit on 120 of each tower's 240 train
photos (/mnt/d/avv/r2r9/models/<T>_ut42/train_png).  If depth is real, rendering the other 120
train views and re-fitting is a cheap, retrain-free upgrade; if it is subset luck, it is nothing.

PRIMARY arms are in PRODUCTION FORM: raw fitted field x 1.30 (what r29 ships).  Two matched-
amplitude arms are carried alongside to show whether depth acts through the field's SIZE or its
STRUCTURE.

Production harness: HCM0181, 60 REAL test poses, REAL test GT.  FULL shipped chain per arm.
"""
import io, os, sys, time, json, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore                       # noqa: E402
from fieldlib import upsample, warp                      # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
FR = os.path.join(HERE, "fieldrefit")
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
ARMS = ["n120", "n90a", "n60fix", "n60a", "n60b", "n30a", "n120m", "n60am"]
REF = "n120"


def ldt(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def report(per, acc, N, out, final):
    res = {}
    for a in ARMS:
        P, S, L = acc[a] / N
        res[a] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                      psnr=P, ssim=S, lpips=L)
    json.dump({"n": N, "final": final, "res": res, "per": per}, open(out, "w"), indent=1)
    if not final:
        return
    print(f"\nFIT-POOL DEPTH, {TAG}, n={N}, FULL shipped chain, fields in PRODUCTION form "
          f"(raw fit x1.30) unless suffixed m")
    print(f"{'arm':>8} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs n120':>9} {'paired t':>9} {'win/N':>7}")
    b = np.array(per[REF])
    for a in ARMS:
        v = np.array(per[a]) - b
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 else float("nan")
        ts = f"{t:9.2f}" if t == t else f"{'--':>9}"
        r_ = res[a]
        print(f"{a:>8} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res[REF]['score']:+9.4f} {ts} {int((v>0).sum()):3d}/{N}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(FR, "res_depth2.json"))
    args = ap.parse_args()
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    dirs = [D(m) for m in POOL]
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    if args.limit:
        stems = stems[:args.limit]

    med = lambda st: np.median(st.astype(np.float32), 0)
    ZB9 = np.load(os.path.join(FR, f"flow_{TAG}_B9.npz"))
    Z11 = np.load(os.path.join(FR, f"flow_{TAG}_B11.npz"))
    ZA = np.load(os.path.join(FR, f"flow_{TAG}_B11all.npz"))
    H, W = [int(x) for x in ZB9["HW"]]
    SA = ZA["s8"]
    NA = SA.shape[0]
    all_stems = [str(s) for s in ZA["stems"]]
    fix_idx = np.array([all_stems.index(s) for s in (str(x) for x in Z11["stems"])
                        if s in all_stems])
    assert len(fix_idx) == 60, len(fix_idx)

    S8 = {"n120": med(SA), "n60fix": med(SA[fix_idx])}
    for tag_, n_, sd in (("n90a", 90, 11), ("n60a", 60, 1), ("n60b", 60, 2), ("n30a", 30, 3)):
        sel = np.random.RandomState(sd).choice(NA, n_, replace=False)
        S8[tag_] = med(SA[sel])

    mag = lambda f: float(np.linalg.norm(f, axis=2).mean())
    b9mag = mag(med(ZB9["s8"]))
    F = {a: S8[a] * 1.30 for a in ("n120", "n90a", "n60fix", "n60a", "n60b", "n30a")}
    F["n120m"] = S8["n120"] * (b9mag * 1.30 / mag(S8["n120"]))
    F["n60am"] = S8["n60a"] * (b9mag * 1.30 / mag(S8["n60a"]))

    print(f"B9 field mean|f| {b9mag:.4f} px (shipped fit target, 60 views)")
    print(f"{'arm':>8} {'mean|f|':>8} {'p95|f|':>8} {'cos(.,n120)':>12} {'mean|f-n120|':>13}")
    for a in ARMS:
        f = F[a]
        c = float((f * F[REF]).sum() /
                  np.sqrt((f * f).sum() * (F[REF] * F[REF]).sum()))
        mg = np.linalg.norm(f, axis=2)
        print(f"{a:>8} {mg.mean():8.4f} {np.percentile(mg,95):8.4f} {c:12.5f} "
              f"{np.linalg.norm(f-F[REF],axis=2).mean():13.4f}", flush=True)
    FU = {a: upsample(F[a], H, W, "cubic") for a in ARMS}

    acc = {a: np.zeros(3) for a in ARMS}
    per = {a: [] for a in ARMS}
    t0 = time.time()
    for n, s in enumerate(stems):
        mem = [ldt(os.path.join(d, s + ".png")) for d in dirs]
        ens = torch.stack(mem).mean(0)
        er = restore(ens, mem, 1.0, len(mem)).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        del mem, ens
        g = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

        def chain(a):
            x = np.clip(warp(er, FU[a], "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                              dtype=np.float32) / 255.0

        with ThreadPoolExecutor(max_workers=8) as ex:
            jj = dict(zip(ARMS, ex.map(chain, ARMS)))
        for a in ARMS:
            r = torch.from_numpy(np.ascontiguousarray(jj[a])).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[a] += (P, S, L)
            per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
            del r
        del g
        if n % 10 == 9 or n == len(stems) - 1:
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(per, acc, n + 1, args.out, final=False)
    report(per, acc, len(stems), args.out, final=True)
    print(f"\nwrote {args.out}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

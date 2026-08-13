"""Phase 6: does the production JPEG encode move the optimal ensemble DEPTH?

Pixel-mean averaging destroys per-view texture; the shipped q100/ss2 JPEG is known to
partly substitute for it. Deeper ensembles are smoother, so the encode could plausibly
pay off more at large k and shift the optimum. Measured, not argued.
Encode settings copied verbatim from build_r26.sh (the shipped r26 builder).
"""
import os, sys, io, json, time
import numpy as np
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:0"
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)

R = np.load(os.path.join(TMP, "renders_u8.npy"), mmap_mode="r")
G = np.load(os.path.join(TMP, "gt_u8.npy"), mmap_mode="r")
N = open(os.path.join(TMP, "names.txt")).read().split()[:21]
p3 = json.load(open(os.path.join(HERE, "p3.json")))
order = [N.index(x) for x in p3["A_order"]]

KS = list(range(2, 13))
cands = {f"rank_k{k}": order[:k] for k in KS}
cands["UT4_shipped"] = [N.index(x) for x in
                        ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]]

acc = {f"{n}|{enc}": {"psnr": [], "ssim": [], "lpips": []}
       for n in cands for enc in ("png", "jpg")}
net = lpips_pkg.LPIPS(net="vgg").to(DEV).eval()
t0 = time.time()
with torch.no_grad():
    for v in range(60):
        M = np.asarray(R[:21, v], dtype=np.float32)
        g = torch.from_numpy(np.ascontiguousarray(G[v])).to(DEV).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        for n, idxs in cands.items():
            e = np.round(M[idxs].mean(0)).clip(0, 255).astype(np.uint8)
            b = io.BytesIO(); Image.fromarray(e).save(b, "JPEG", **SHIPPED)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
            for enc, arr in (("png", e), ("jpg", j)):
                r = torch.from_numpy(arr).to(DEV).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                mse = float(((r - g) ** 2).mean())
                k = f"{n}|{enc}"
                acc[k]["psnr"].append(10 * np.log10(1.0 / max(mse, 1e-12)))
                acc[k]["ssim"].append(float(repo_ssim(r, g)))
                acc[k]["lpips"].append(float(net(r * 2 - 1, g * 2 - 1).item()))
                del r
        del M, g
        torch.cuda.empty_cache()
        if v % 10 == 0:
            print("view", v, "%.0fs" % (time.time() - t0), flush=True)

json.dump(acc, open(os.path.join(HERE, "p6.json"), "w"))


def agg(k):
    a = acc[k]
    P_, S_, L_ = np.mean(a["psnr"]), np.mean(a["ssim"]), np.mean(a["lpips"])
    return 100 * (0.4 * (1 - L_) + 0.3 * S_ + 0.3 * min(P_ / 50, 1)), P_, S_, L_


print(f"\n{'cand':14s} {'PNG':>9s} {'JPEG q100ss2':>13s} {'jpeg-png':>9s}")
for n in list(cands):
    sp = agg(f"{n}|png")[0]; sj = agg(f"{n}|jpg")[0]
    print(f"{n:14s} {sp:9.4f} {sj:13.4f} {sj-sp:+9.4f}")
print("elapsed %.0fs" % (time.time() - t0))

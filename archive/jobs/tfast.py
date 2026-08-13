import os, sys, json, time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
torch.set_num_threads(12)
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
vgg = lpips_pkg.LPIPS(net="vgg").eval()
D = "/mnt/d/avv"; PUB = f"{D}/data/phase1/public_set"
PAIRS = [
 ("ES_chair",  f"{D}/chair_eval/lpearly60k/eval_png", f"{D}/evalsplit/chair/eval_gt", "ES"),
 ("ES_bonsai", f"{D}/bonsai_perc/pC_lpearly/eval_png", f"{D}/evalsplit/bonsai/eval_gt", "ES"),
 ("ES_0421",   f"{D}/evalgen/HCM0421/eval_png", f"{D}/evalsplit/HCM0421/eval_gt", "ES"),
 ("ES_0181",   f"{D}/lpsweep/HCM0181_lp0.3/eval_png", f"{D}/evalsplit/HCM0181/eval_gt", "ES"),
 ("0181_SHIP", f"{D}/prodharness/k4f", f"{PUB}/HCM0181/test/images", "PROD-chain"),
 ("0181_solo", f"{D}/output/HCM0181_gsplatB9ut/test_poses_renders_png", f"{PUB}/HCM0181/test/images", "PROD-solo"),
 ("0193_solo", f"{D}/output/HCM0193_gsplatB9ut/test_poses_renders_png", f"{PUB}/HCM0193/test/images", "PROD-solo"),
 ("0204_solo", f"{D}/output/HCM0204_gsplatB9ut/test_poses_renders_png", f"{PUB}/HCM0204/test/images", "PROD-solo"),
 ("0031_solo", f"{D}/output/hcm0031_gsplatB9ut/test_poses_renders_png", f"{PUB}/hcm0031/test/images", "PROD-solo"),
 ("0034_solo", f"{D}/output/hcm0034_gsplatB9ut/test_poses_renders_png", f"{PUB}/hcm0034/test/images", "PROD-solo"),
]
NL = 8
def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.
                            ).permute(2, 0, 1).unsqueeze(0)
out = []
for tag, rd, gd, reg in PAIRS:
    if not (os.path.isdir(rd) and os.path.isdir(gd)):
        print("MISSING", tag, flush=True); continue
    gt = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
    rn = {os.path.splitext(f)[0]: f for f in os.listdir(rd)
          if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
    st = sorted(set(gt) & set(rn)); n = len(st)
    if n == 0:
        print("EMPTY", tag, flush=True); continue
    idx = set(np.linspace(0, n - 1, min(NL, n)).astype(int).tolist())
    P = S = L = 0.0; nl = 0; t0 = time.time(); bad = False
    with torch.no_grad():
        for i, s in enumerate(st):
            r = load(os.path.join(rd, rn[s])); g = load(os.path.join(gd, gt[s]))
            if r.shape != g.shape:
                print("SHAPE", tag, s, tuple(r.shape), tuple(g.shape), flush=True); bad = True; break
            P += 10 * np.log10(1. / max(((r - g) ** 2).mean().item(), 1e-12))
            S += float(repo_ssim(r, g))
            if i in idx:
                L += float(vgg(r * 2 - 1, g * 2 - 1).item()); nl += 1
    if bad:
        continue
    P /= n; S /= n; L /= max(nl, 1)
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.))
    rec = dict(tag=tag, regime=reg, n=n, n_lpips=nl, gt_only=len(gt) - n, rn_only=len(rn) - n,
               psnr=P, ssim=S, lpips=L, score=sc, secs=round(time.time() - t0, 1))
    out.append(rec)
    print(f"RES {tag:10s} {reg:11s} n={n:3d} nL={nl} PSNR {P:8.4f} SSIM {S:.4f} "
          f"LPIPS {L:.4f} SCORE {sc:.4f} ({rec['secs']}s)", flush=True)
    json.dump(out, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/tfast.json", "w"), indent=1)
print("DONE", flush=True)

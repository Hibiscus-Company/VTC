"""(c) PSNR clamp check + establish harness baselines."""
import sys, json
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from harness import *

rows = []
print(f"{'set':28s} {'n':>3s} {'PSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s} {'SCORE':>8s} {'maxPSNR':>8s}")
for s in SCENES:
    d = metrics(R_TEST.format(s=s), GT_TEST.format(s=s), per_image=True)
    mx = max(d["psnr_list"])
    rows.append((f"{s}/B9ut", d, mx))
    print(f"{s+'/B9ut':28s} {d['n']:3d} {d['PSNR']:8.4f} {d['SSIM']:7.4f} {d['LPIPS']:7.4f} {d['SCORE']:8.4f} {mx:8.3f}")

d = metrics(K4, GT_TEST.format(s="HCM0181"), per_image=True)
mx = max(d["psnr_list"])
rows.append(("HCM0181/k4", d, mx))
print(f"{'HCM0181/k4':28s} {d['n']:3d} {d['PSNR']:8.4f} {d['SSIM']:7.4f} {d['LPIPS']:7.4f} {d['SCORE']:8.4f} {mx:8.3f}")

json.dump({k: {kk: vv for kk, vv in v.items()} | {"maxPSNR": m} for k, v, m in rows},
          open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/baseline.json", "w"), indent=1)

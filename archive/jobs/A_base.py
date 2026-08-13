"""Baselines on the production harness: members, k4, k4+field; and all 5 scenes single-member."""
import os, sys, json
import numpy as np, torch, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(8); torch.set_num_threads(8)

dev = "cuda"
lp = mlib.LP(dev)
FLD = {s: np.load(f"/mnt/d/avv/fields/{s}.npy") for s in
       ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]}


def sc(imgs8, gts8):
    P = S = L = 0.0
    for a, g in zip(imgs8, gts8):
        ta, tg = mlib.to_t(a, dev), mlib.to_t(g, dev)
        P += mlib.psnr(ta, tg); S += float(mlib.ssim(ta, tg)); L += lp(ta, tg)
    n = len(imgs8)
    P, S, L = P / n, S / n, L / n
    return P, S, L, mlib.score(P, S, L)


st = stems()
gd, gmap = gt_map("HCM0181")
GT = [mlib.load_u8(os.path.join(gd, gmap[s])) for s in st]
res = {}
for m in MEM:
    R = [mlib.load_u8(os.path.join(ROOT, m, "test_poses_renders_png", s + ".png")) for s in st]
    res[m] = sc(R, GT)
    print(m, "%.4f %.5f %.5f  score %.4f" % res[m], flush=True)

K = [mlib.load_u8(os.path.join(K4, s + ".png")) for s in st]
res["k4"] = sc(K, GT); print("k4", "%.4f %.5f %.5f  score %.4f" % res["k4"], flush=True)
KF = [apply_field(k, FLD["HCM0181"]) for k in K]
res["k4+field"] = sc(KF, GT); print("k4+field", "%.4f %.5f %.5f  score %.4f" % res["k4+field"], flush=True)
os.makedirs("/mnt/d/avv/prodharness/k4f", exist_ok=True)
from PIL import Image
for s, a in zip(st, KF):
    Image.fromarray(a).save(f"/mnt/d/avv/prodharness/k4f/{s}.png")

print("\n=== per-scene single member gsplatB9ut ===", flush=True)
for scene in ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]:
    d = f"/mnt/d/avv/output/{scene}_gsplatB9ut/test_poses_renders_png"
    gdir, gm = gt_map(scene)
    ss = [os.path.splitext(f)[0] for f in sorted(os.listdir(d))]
    R = [mlib.load_u8(os.path.join(d, s + ".png")) for s in ss]
    G = [mlib.load_u8(os.path.join(gdir, gm[s])) for s in ss]
    a = sc(R, G)
    RF = [apply_field(r, FLD[scene]) for r in R]
    b = sc(RF, G)
    res["B9_" + scene] = a; res["B9f_" + scene] = b
    print("%-9s raw  %.4f %.5f %.5f score %.4f | +field %.4f %.5f %.5f score %.4f (%+.4f)"
          % ((scene,) + a + b + (b[3] - a[3],)), flush=True)

json.dump({k: list(v) for k, v in res.items()},
          open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/A_base.json", "w"), indent=1)

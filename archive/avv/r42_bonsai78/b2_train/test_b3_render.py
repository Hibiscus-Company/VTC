#!/usr/bin/env python
"""CPU-only tests for the B2 (3) renderer path. NO GPU."""
import os, sys, csv, json, types
import numpy as np
import torch

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/d/avv/r42_bonsai78/b2_train")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rb2", "/mnt/d/avv/r42_bonsai78/b2_train/render_gsplat_b2.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
import train_gsplat_b2 as T

ok = True


def chk(n, c, e=""):
    global ok
    ok &= bool(c)
    print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")


print("1. train/test operator PARITY: gauss_blur_np == gauss_blur1")
img = np.random.RandomState(0).rand(48, 72, 3)
for s in (0.4, 1.0, 2.3):
    a = R.gauss_blur_np(img, s, 8)
    b = T.gauss_blur1(torch.from_numpy(img), torch.tensor(s, dtype=torch.float64), 8).numpy()
    m = float(np.abs(a - b).max())
    chk(f"sigma={s} identical to the trainer's kernel", m < 1e-12, f"max|d| {m:.2e}")
chk("sigma<=1e-3 is a strict no-op", R.gauss_blur_np(img, 0.0, 8) is img)

print("2. frame_index parsing")
for n, want in (("frame_002190.jpg", 2190), ("frame_000010.png", 10),
                ("DJI_20241229103156_0001_V.JPG", 20241229103156)):
    chk(f"{n} -> {R.frame_index(n)}", R.frame_index(n) == want)
chk("unparseable -> None", R.frame_index("abc.jpg") is None)

print("3. build_sigma_lookup on the REAL bonsai eval split")
side = {r["name"]: float(r["sigma_init"]) for r in
        csv.DictReader(open("/mnt/d/avv/r42_bonsai78/b2_train/bonsai_sharp_sidecar.csv"))}
EV = set(json.load(open("/mnt/d/avv/evalsplit/bonsai/split.json"))["eval_frames"])
tr_names = [n for n in sorted(side) if int(n[6:12]) not in EV]
blur = {"sigma": torch.tensor([side[n] for n in tr_names]), "names": tr_names,
        "sigma_max": 2.5, "radius": 8}
rows = list(csv.DictReader(open("/mnt/d/avv/evalsplit/bonsai/eval_poses.csv")))
chk("28 eval rows", len(rows) == 28, str(len(rows)))
for gam in (0.0, 0.6, 1.0, 1.4):
    lut, how = R.build_sigma_lookup(blur, rows, "predict", gam, 13.0, 2.5, 0.0)
    pred = np.array([lut[r["image_name"]] for r in rows])
    true = np.array([side[r["image_name"]] for r in rows])
    r2 = 1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()
    print(f"    gamma={gam:<4} how={how:9s} pred med {np.median(pred):.3f} sd {pred.std():.4f} "
          f"| vs the hole's OWN sigma_init: R2 {r2:+.3f} rmse {np.sqrt(((pred-true)**2).mean()):.3f} px "
          f"corr {np.corrcoef(pred, true)[0,1]:+.3f}")
    chk(f"  gamma={gam} within [0,2.5]", (pred >= 0).all() and (pred <= 2.5).all())
lut, _ = R.build_sigma_lookup(blur, rows, "predict", 1.0, 13.0, 2.5, 0.0)
pred = np.array([lut[r["image_name"]] for r in rows])
true = np.array([side[r["image_name"]] for r in rows])
trs = np.asarray(blur["sigma"], dtype=np.float64)
print(f"    SHRINK: pred sd {pred.std():.4f} vs train sd {trs.std():.4f} "
      f"= {pred.std()/trs.std():.2f}x  (a kernel average is shrunk by construction)")
chk("gamma=0 collapses to zero everywhere",
    max(R.build_sigma_lookup(blur, rows, "predict", 0.0, 13.0, 2.5, 0.0)[0].values()) == 0.0)
lutc, howc = R.build_sigma_lookup(blur, rows, "const", 1.0, 13.0, 2.5, 0.7)
chk("const mode is constant", howc == "const" and len(set(lutc.values())) == 1,
    f"{list(lutc.values())[0]:.3f}")

print("4. bandwidth sensitivity of the sigma prediction (h in frames)")
for h in (7.0, 13.0, 20.0, 40.0):
    lut, _ = R.build_sigma_lookup(blur, rows, "predict", 1.0, h, 2.5, 0.0)
    p = np.array([lut[r["image_name"]] for r in rows])
    r2 = 1 - ((p - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()
    print(f"    h={h:<5} R2 {r2:+.3f}  sd {p.std():.4f}  rmse {np.sqrt(((p-true)**2).mean()):.3f} px")

print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)

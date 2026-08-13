#!/usr/bin/env python
"""REFUTATION TEST 1 -- is the field-magnitude shrink a MODEL-CLASS effect or a SCENE effect?

The claim: alpha (the field amplitude gain) is a property of the FIT MEMBER's convergence.
Evidence offered: on HCM0181, |f(B11 60k/8M)| / |f(B9 30k/5M)| = 0.827, and
                  |f(private ut42)| / |f(public B9)| = 0.823 -> "0.4% match" -> same mechanism.

That inference is only valid if 30k/5M -> 60k/8M really shrinks |f_train| by ~0.83 IN GENERAL.
We can test that DIRECTLY and with n=5 on the very scenes that ship:
   /mnt/d/avv/r2r8/models/<T>_ut42  = 30k iters / 5M cap   (exactly the B9 class)
   /mnt/d/avv/r2r9/models/<T>_ut42  = 60k iters / 8M cap   (exactly the B11 class)
same 5 private towers, same TRAIN photos, same 120 stems, same ds=8, same MEDIAN estimator.
If the ratio is ~0.83 the claim's mechanism is real. If it is ~1.0 the 0.823 private/public
ratio is a SCENE property and re-scaling the private gain by 1/0.83 is unfounded.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import cv2
cv2.setNumThreads(2)
from gsplat_track.fit_field import fit_field

TOWERS = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]
DATA = "/mnt/d/avv/data/phase1/private_set2"
OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/refute"
os.makedirs(OUT, exist_ok=True)

res = {}
t0 = time.time()
for T in TOWERS:
    gt = f"{DATA}/{T}/train/images"
    row = {}
    for cls, root in (("30k5M", "/mnt/d/avv/r2r8"), ("60k8M", "/mnt/d/avv/r2r9")):
        rd = f"{root}/models/{T}_ut42/train_png"
        f = fit_field(rd, gt, ds=8, estimator="median", verbose=False)
        np.save(os.path.join(OUT, f"{T}_{cls}.npy"), f.astype(np.float32))
        m = np.linalg.norm(f, axis=2)
        row[cls] = float(m.mean())
        row[cls + "_max"] = float(m.max())
    # shape correlation between the two classes' fields
    a = np.load(os.path.join(OUT, f"{T}_30k5M.npy")).ravel()
    b = np.load(os.path.join(OUT, f"{T}_60k8M.npy")).ravel()
    row["corr"] = float(np.corrcoef(a, b)[0, 1])
    row["ratio_60k_over_30k"] = row["60k8M"] / row["30k5M"]
    # least-squares scalar that best maps the 60k field onto the 30k field
    row["ls_scale_30k_from_60k"] = float((a @ b) / (b @ b))
    res[T] = row
    print(f"{T}  30k5M {row['30k5M']:.4f}  60k8M {row['60k8M']:.4f}  "
          f"ratio {row['ratio_60k_over_30k']:.4f}  LS {row['ls_scale_30k_from_60k']:.4f}  "
          f"corr {row['corr']:.4f}   [{time.time()-t0:.0f}s]", flush=True)

r = np.array([res[T]["ratio_60k_over_30k"] for T in TOWERS])
ls = np.array([res[T]["ls_scale_30k_from_60k"] for T in TOWERS])
print(f"\nratio 60k/30k  mean {r.mean():.4f}  sd {r.std(ddof=1):.4f}  "
      f"min {r.min():.4f}  max {r.max():.4f}")
print(f"LS scale       mean {ls.mean():.4f}  sd {ls.std(ddof=1):.4f}")
print(f"CLAIM PREDICTS ratio ~= 0.827")
res["_summary"] = dict(ratio_mean=float(r.mean()), ratio_sd=float(r.std(ddof=1)),
                       ls_mean=float(ls.mean()), claim=0.827)
json.dump(res, open(os.path.join(OUT, "res_class.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "res_class.json"))

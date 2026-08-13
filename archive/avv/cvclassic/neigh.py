#!/usr/bin/env python
"""Is the frame-specific residual warp field a smooth function of VIEWPOINT?

If yes, the field measured at TRAIN photos can be interpolated to a test pose
(legal, no test GT) -- a 1st-order generalisation of the shipped global lens field.
If no (fields uncorrelated between neighbouring views), the +1.58 oracle is either
flow-overfitting to GT or genuinely unreachable.

Correlation between per-frame residual fields, as a function of pose separation.
Doubles as an OVERFIT TEST: DIS-flow noise fitted to GT cannot correlate across frames.
"""
import os, csv, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="/tmp/cvdiag/res")
ap.add_argument("--poses", default="/mnt/d/avv/evalsplit/HCM0181/eval_poses.csv")
a = ap.parse_args()

F = np.load(os.path.join(a.res, "fields.npy"))
mu = np.load(os.path.join(a.res, "field_mean.npy"))
R = (F - mu).reshape(len(F), -1)
R -= R.mean(0, keepdims=True)

rows = list(csv.DictReader(open(a.poses)))
print("csv cols:", list(rows[0]))
names = [os.path.splitext(r["image_name"])[0] for r in rows]
order = np.argsort(names)
q = np.array([[float(r[k]) for k in ("qw", "qx", "qy", "qz")] for r in rows])[order]
t = np.array([[float(r[k]) for k in ("tx", "ty", "tz")] for r in rows])[order]
# camera centre C = -R^T t
def qrot(qq):
    w, x, y, z = qq
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
C = np.array([-qrot(q[i]).T @ t[i] for i in range(len(q))])
# viewing direction = R^T [0,0,1]
D = np.array([qrot(q[i]).T @ np.array([0, 0, 1.0]) for i in range(len(q))])

n = min(len(R), len(C))
R, C, D = R[:n], C[:n], D[:n]
Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
COR = Rn @ Rn.T
ang = np.degrees(np.arccos(np.clip(D @ D.T, -1, 1)))
dist = np.linalg.norm(C[:, None] - C[None], axis=2)

iu = np.triu_indices(n, 1)
c, ag, ds = COR[iu], ang[iu], dist[iu]
print(f"\nn={n} eval frames. Correlation of the frame-specific residual field vs view separation")
edges = [0, 2, 4, 6, 8, 12, 20, 40, 200]
print(f"{'view angle sep':>16} {'pairs':>6} {'mean corr':>10} {'p90 corr':>9}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (ag >= lo) & (ag < hi)
    if m.sum() > 2:
        print(f"{lo:6.0f}-{hi:<9.0f} {m.sum():6d} {c[m].mean():10.4f} {np.quantile(c[m],0.9):9.4f}")
print(f"\noverall mean corr {c.mean():+.4f}   max {c.max():.4f}")
k = np.argsort(-c)[:8]
print("top pairs (corr, angle deg, baseline):")
for i in k:
    print(f"   {c[i]:.4f}  {ag[i]:7.2f} deg  {ds[i]:.4f}")
# nearest-neighbour predictability: how much of frame i's field does its nearest view explain?
nn = np.array([np.argmin(np.where(np.arange(n) == i, 1e9, ang[i])) for i in range(n)])
print(f"\nnearest-view: mean angle {ang[np.arange(n),nn].mean():.2f} deg, "
      f"mean field corr {COR[np.arange(n),nn].mean():+.4f}  "
      f"(=> R^2 {np.mean(COR[np.arange(n),nn]**2)*100:.2f}% of the residual field)")

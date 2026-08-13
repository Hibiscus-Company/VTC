#!/usr/bin/env python
"""Bonsai error decomposition: spatial (semantic masks) + view-dependence + blur + LPIPS masking."""
import os, sys, json, csv, math
import numpy as np, cv2, torch

GT   = "/mnt/d/avv/evalsplit/bonsai/eval_gt"          # 28 held-out real photos
RD   = "/mnt/d/avv/bonsai_perc/pC_lpearly/eval_png"   # shipped-recipe eval render (out-of-sample)
TRG  = "/mnt/d/avv/blurbound/bonsai/train_png"        # 220 train photos (in-sample GT)
TRR  = "/mnt/d/avv/blurbound/bonsai/train_render"     # in-sample render
POS  = "/mnt/d/avv/evalsplit/bonsai/eval_poses.csv"
OUT  = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/bview"
os.makedirs(OUT, exist_ok=True)

def rd(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)

def masks(g):
    """g: HxWx3 float RGB 0-255 GT. returns dict of bool masks."""
    gray = g.mean(2)
    R, G, B = g[..., 0], g[..., 1], g[..., 2]
    # local texture energy: std in 9x9
    m = cv2.blur(gray, (9, 9)); m2 = cv2.blur(gray * gray, (9, 9))
    tex = np.sqrt(np.maximum(m2 - m * m, 0))
    foliage = (G - np.maximum(R, B) > 8)
    # glass table: dark + smooth -> largest filled component
    cand = ((gray < 115) & (tex < 12)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n > 1:
        k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        tab = (lab == k).astype(np.uint8)
    else:
        tab = cand
    # fill holes (plant/pot sitting on it) via convex hull of the component
    cnts, _ = cv2.findContours(tab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(tab)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.fillConvexPoly(hull, cv2.convexHull(c), 1)
    table = hull.astype(bool) & ~foliage
    rest = ~table & ~foliage
    # split rest by texture: carpet is the high-texture stochastic weave, couch/wall low-texture
    hi = rest & (tex >= 14)
    lo = rest & (tex < 14)
    return dict(table=table, foliage=foliage, rest_hi=hi, rest_lo=lo), tex

def varlap(g):
    return cv2.Laplacian(g.mean(2), cv2.CV_32F, ksize=3).var()

# ---------------- pass 1: masks + L1 + blur, eval (out-of-sample) ----------------
names = sorted(os.listdir(GT))
rows = []
acc = None
for nm in names:
    g = rd(os.path.join(GT, nm))
    r = rd(os.path.join(RD, nm.replace('.jpg', '.png')))
    M, tex = masks(g)
    e = np.abs(g - r).mean(2)
    rec = dict(name=nm, mae=float(e.mean()), vl_gt=float(varlap(g)), vl_rd=float(varlap(r)))
    for k, m in M.items():
        rec[k + '_frac'] = float(m.mean())
        rec[k + '_mae'] = float(e[m].mean()) if m.any() else 0.0
        rec[k + '_share'] = float(e[m].sum() / e.sum())
    # texture-decile binning (objective, mask-free)
    q = np.quantile(tex, np.linspace(0, 1, 11))
    dec = []
    for i in range(10):
        sel = (tex >= q[i]) & (tex <= q[i + 1] if i == 9 else tex < q[i + 1])
        dec.append(float(e[sel].mean()))
    rec['tex_dec'] = dec
    rows.append(rec)
    if acc is None:
        acc = e.copy()
        cv2.imwrite(f"{OUT}/mask_vis.png", cv2.resize(np.stack([
            (M['table'] * 255).astype(np.uint8), (M['rest_hi'] * 255).astype(np.uint8),
            (M['foliage'] * 255).astype(np.uint8)], -1), (960, 540)))
    else:
        acc += e
json.dump(rows, open(f"{OUT}/eval_rows.json", 'w'))
cv2.imwrite(f"{OUT}/err_mean.png", cv2.applyColorMap(
    np.clip(cv2.resize(acc / len(names), (960, 540)) * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO))

def agg(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}

ks = ['mae', 'vl_gt', 'vl_rd'] + [f'{a}_{b}' for a in ['table', 'foliage', 'rest_hi', 'rest_lo'] for b in ['frac', 'mae', 'share']]
print("EVAL (held-out, n=%d):" % len(rows))
for k, v in agg(rows, ks).items():
    print(f"  {k:18s} {v:.4f}")
print("  tex_dec_mae      ", np.round(np.mean([r['tex_dec'] for r in rows], 0), 2).tolist())

# ---------------- pass 2: in-sample (train views) same decomposition ----------------
tn = sorted(os.listdir(TRG))
sel = tn[::14][:16]
trows = []
for nm in sel:
    g = rd(os.path.join(TRG, nm))
    rp = os.path.join(TRR, nm.replace('.png', '.jpg'))
    if not os.path.exists(rp):
        rp = os.path.join(TRR, nm)
    r = rd(rp)
    M, tex = masks(g)
    e = np.abs(g - r).mean(2)
    rec = dict(name=nm, mae=float(e.mean()), vl_gt=float(varlap(g)), vl_rd=float(varlap(r)))
    for k, m in M.items():
        rec[k + '_frac'] = float(m.mean())
        rec[k + '_mae'] = float(e[m].mean()) if m.any() else 0.0
        rec[k + '_share'] = float(e[m].sum() / e.sum())
    trows.append(rec)
json.dump(trows, open(f"{OUT}/train_rows.json", 'w'))
print("IN-SAMPLE TRAIN (n=%d):" % len(trows))
for k, v in agg(trows, ks).items():
    print(f"  {k:18s} {v:.4f}")

# ---------------- pass 3: view geometry ----------------
P = {r['image_name']: r for r in csv.DictReader(open(POS))}
def qR(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
ups, fwd, pos = [], [], []
for nm in names:
    p = P[nm]
    q = [float(p[k]) for k in ('qw', 'qx', 'qy', 'qz')]
    t = np.array([float(p[k]) for k in ('tx', 'ty', 'tz')])
    R = qR(q)
    ups.append(R.T @ np.array([0, -1, 0.]))   # world "up" as seen by cam
    fwd.append(R.T @ np.array([0, 0, 1.]))
    pos.append(-R.T @ t)
up = np.mean(ups, 0); up /= np.linalg.norm(up)
inc = np.array([math.degrees(math.acos(np.clip(-np.dot(f, up), -1, 1))) for f in fwd])  # angle of view dir below horizontal-ish
mae_t = np.array([r['table_mae'] for r in rows]); mae_h = np.array([r['rest_hi'] for r in rows] if False else [r['rest_hi_mae'] for r in rows])
mae_a = np.array([r['mae'] for r in rows])
def cc(a, b): return float(np.corrcoef(a, b)[0, 1])
print("VIEW GEOMETRY: up=", np.round(up, 3).tolist(), " incidence deg mean %.1f sd %.1f range %.1f-%.1f" % (inc.mean(), inc.std(), inc.min(), inc.max()))
print("  corr(incidence, table_mae) = %.3f ; corr(incidence, rest_hi_mae) = %.3f ; corr(incidence, mae) = %.3f" % (cc(inc, mae_t), cc(inc, mae_h), cc(inc, mae_a)))
fi = np.array([int(n.split('_')[1].split('.')[0]) for n in names], float)
vg = np.array([r['vl_gt'] for r in rows]); vr = np.array([r['vl_rd'] for r in rows])
print("  corr(frame_idx, vl_gt) = %.3f ; corr(frame_idx, mae) = %.3f ; corr(vl_gt, mae) = %.3f" % (cc(fi, vg), cc(fi, mae_a), cc(vg, mae_a)))
print("  vl_gt mean %.1f sd %.1f min %.1f max %.1f | vl_rd mean %.1f sd %.1f | ratio rd/gt mean %.3f" % (
    vg.mean(), vg.std(), vg.min(), vg.max(), vr.mean(), vr.std(), float(np.mean(vr / vg))))
np.save(f"{OUT}/inc.npy", inc)

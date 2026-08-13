#!/usr/bin/env python
import os, json, math, csv
import numpy as np, cv2, torch, lpips
torch.set_num_threads(24)
sys_out = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/bview"
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
RD = "/mnt/d/avv/bonsai_perc/pC_lpearly/eval_png"
TRPH = "/mnt/d/avv/data/phase1/private_set2/bonsai/train/images"   # real train photos
TRRD = "/mnt/d/avv/blurbound/bonsai/train_png"                     # in-sample renders (220)

def rd(p):
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)

def masks(g):
    gray = g.mean(2); R, G, B = g[..., 0], g[..., 1], g[..., 2]
    m = cv2.blur(gray, (9, 9)); m2 = cv2.blur(gray * gray, (9, 9))
    tex = np.sqrt(np.maximum(m2 - m * m, 0))
    foliage = (G - np.maximum(R, B) > 8)
    cand = ((gray < 115) & (tex < 12)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    tab = (lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) if n > 1 else cand
    cnts, _ = cv2.findContours(tab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(tab)
    if cnts:
        cv2.fillConvexPoly(hull, cv2.convexHull(max(cnts, key=cv2.contourArea)), 1)
    table = hull.astype(bool) & ~foliage
    rest = ~table & ~foliage
    return dict(table=table, foliage=foliage, rest_hi=rest & (tex >= 14), rest_lo=rest & (tex < 14)), tex

def vl(g): return float(cv2.Laplacian(g.mean(2), cv2.CV_32F, ksize=3).var())

# ---------- A. TRUE in-sample fit: train photos vs in-sample renders ----------
print("== A. IN-SAMPLE (train photo vs its own render) vs HELD-OUT ==", flush=True)
tnames = sorted(os.listdir(TRRD))[::14][:16]
rowsA = []
for nm in tnames:
    ph = os.path.join(TRPH, nm.replace('.png', '.jpg'))
    if not os.path.exists(ph): continue
    g = rd(ph); r = rd(os.path.join(TRRD, nm))
    M, tex = masks(g); e = np.abs(g - r).mean(2)
    rec = dict(mae=float(e.mean()), vl_gt=vl(g), vl_rd=vl(r))
    for k, m in M.items():
        rec[k + '_frac'] = float(m.mean()); rec[k + '_mae'] = float(e[m].mean()) if m.any() else 0.
        rec[k + '_share'] = float(e[m].sum() / e.sum())
    rowsA.append(rec)
for k in sorted(rowsA[0]):
    print("  IN %-16s %.4f" % (k, np.mean([r[k] for r in rowsA])), flush=True)

# ---------- B. temporal-neighbour blur test on GT ----------
print("== B. GT MOTION BLUR (eval GT vs its +-10 frame train-photo neighbours) ==", flush=True)
allph = sorted(os.listdir(TRPH))
idx = {int(n.split('_')[1].split('.')[0]): n for n in allph}
enames = sorted(os.listdir(GT))
ratios = []
for nm in enames:
    i = int(nm.split('_')[1].split('.')[0])
    nb = [idx[j] for j in (i - 10, i + 10) if j in idx]
    if not nb: continue
    v0 = vl(rd(os.path.join(GT, nm)))
    vn = np.mean([vl(rd(os.path.join(TRPH, x))) for x in nb])
    ratios.append(v0 / vn)
ratios = np.array(ratios)
print("  varLap(evalGT)/varLap(neighbours): mean %.3f  median %.3f  min %.3f max %.3f  n=%d" %
      (ratios.mean(), np.median(ratios), ratios.min(), ratios.max(), len(ratios)), flush=True)
vgs = np.array([vl(rd(os.path.join(TRPH, n))) for n in allph])
print("  varLap over ALL 248 train photos: mean %.0f sd %.0f min %.0f max %.0f  p10 %.0f p90 %.0f" %
      (vgs.mean(), vgs.std(), vgs.min(), vgs.max(), np.percentile(vgs, 10), np.percentile(vgs, 90)), flush=True)

# ---------- C. LPIPS spatial decomposition + region counterfactuals ----------
print("== C. LPIPS (vgg) SPATIAL DECOMPOSITION, held-out n=28 ==", flush=True)
L = lpips.LPIPS(net='vgg', spatial=True)
T = lambda x: torch.from_numpy(x / 255.).permute(2, 0, 1)[None].float() * 2 - 1
out = []
for nm in enames:
    g = rd(os.path.join(GT, nm)); r = rd(os.path.join(RD, nm.replace('.jpg', '.png')))
    M, tex = masks(g)
    with torch.no_grad():
        sp = L(T(g), T(r))[0, 0].numpy()
    rec = dict(name=nm, lpips=float(sp.mean()))
    for k, m in M.items():
        rec[k + '_frac'] = float(m.mean())
        rec[k + '_contrib'] = float(sp[m].sum() / sp.size)      # additive share of LPIPS
        rec[k + '_dens'] = float(sp[m].mean()) if m.any() else 0.
    # counterfactual: paste GT into region -> LPIPS of the "perfect there" image
    for k in ('table', 'rest_hi'):
        rr = r.copy(); rr[M[k]] = g[M[k]]
        with torch.no_grad():
            rec[k + '_cf'] = float(L(T(g), T(rr)).mean())
    # texture-decile LPIPS density
    q = np.quantile(tex, np.linspace(0, 1, 11)); dec = []
    for i in range(10):
        s = (tex >= q[i]) & ((tex <= q[i + 1]) if i == 9 else (tex < q[i + 1]))
        dec.append(float(sp[s].mean()) if s.any() else 0.)
    rec['tex_dec'] = dec
    out.append(rec)
    print("   %s lpips %.4f table %.4f/%.4f hi %.4f/%.4f" % (nm, rec['lpips'], rec['table_contrib'], rec['table_cf'], rec['rest_hi_contrib'], rec['rest_hi_cf']), flush=True)
json.dump(out, open(f"{sys_out}/lpips_rows.json", 'w'))
print("== LPIPS SUMMARY ==", flush=True)
for k in ['lpips'] + [f'{a}_{b}' for a in ['table', 'foliage', 'rest_hi', 'rest_lo'] for b in ['frac', 'contrib', 'dens']] + ['table_cf', 'rest_hi_cf']:
    print("  %-18s %.5f" % (k, np.mean([r[k] for r in out])), flush=True)
print("  tex_dec_lpips", np.round(np.mean([r['tex_dec'] for r in out], 0), 4).tolist(), flush=True)

"""R2: where is the squared error?  Concentration + region masks."""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
H, W = 989, 1320
N = 60

qs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
cum = np.zeros(len(qs))
gini_se = []

# region accumulators
REG = ["sky", "flat_nonsky", "midtex", "hightex", "edge_near"]
se_sum = {r: 0.0 for r in REG}
px_sum = {r: 0.0 for r in REG}
grad_dec_se = np.zeros(10); grad_dec_px = np.zeros(10)
tot_se = 0.0

masks_all = np.zeros((N, H, W), np.uint8)   # region code
se_all = np.zeros((N, H, W), np.float32)

for i in range(N):
    g = G[i].astype(np.float32) / 255.0
    p = apply_field(R[k4i, i].astype(np.float32) / 255.0, field)
    se = ((p - g) ** 2).mean(2)
    se_all[i] = se
    tot_se += se.sum()
    v = np.sort(se.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    for j, q in enumerate(qs):
        cum[j] += c[int(q * v.size) - 1]

    y = (g * np.array([0.299, 0.587, 0.114], np.float32)).sum(2)
    ys = cv2.GaussianBlur(y, (0, 0), 1.0)
    gx = cv2.Sobel(ys, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ys, cv2.CV_32F, 0, 1, ksize=3)
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.0)

    # gradient deciles
    thr = np.quantile(gm, np.linspace(0, 1, 11)[1:-1])
    dec = np.digitize(gm, thr)
    for d in range(10):
        m = dec == d
        grad_dec_se[d] += se[m].sum(); grad_dec_px[d] += m.sum()

    # sky: smooth + bright + connected to top
    smooth = gm < np.quantile(gm, 0.35)
    bright = y > 0.45
    cand = (smooth & bright).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab = cv2.connectedComponents(cand)
    top = np.unique(lab[:8][cand[:8] > 0])
    sky = np.isin(lab, top[top > 0]) if top.size else np.zeros_like(cand, bool)
    sky = cv2.dilate(sky.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    # strong edges & their neighbourhood
    strong = gm > np.quantile(gm, 0.92)
    near = cv2.dilate(strong.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    code = np.zeros((H, W), np.uint8)
    hightex = (~sky) & (gm >= np.quantile(gm, 0.85))
    midtex = (~sky) & (~hightex) & (gm >= np.quantile(gm, 0.45))
    flatn = (~sky) & (gm < np.quantile(gm, 0.45))
    for k, m in [("sky", sky), ("flat_nonsky", flatn), ("midtex", midtex),
                 ("hightex", hightex), ("edge_near", near)]:
        se_sum[k] += se[m].sum(); px_sum[k] += m.sum()
    code[sky] = 1; code[flatn] = 2; code[midtex] = 3; code[hightex] = 4
    masks_all[i] = code

np.save(OUT + "/se_all.npy", se_all)
np.save(OUT + "/code_all.npy", masks_all)

cum /= N
print("== SE concentration (post-field k4) ==")
print(" top-frac-of-pixels   share-of-total-SE")
for q, c in zip(qs, cum):
    print(f"   {100*q:6.2f}%            {100*c:7.2f}%")

print("\n== SE by GT-gradient decile (0=flattest) ==")
print(" dec   px%     SE%    meanSE/overall")
for d in range(10):
    print(f"  {d}   {100*grad_dec_px[d]/grad_dec_px.sum():5.1f}  {100*grad_dec_se[d]/grad_dec_se.sum():6.2f}"
          f"   {(grad_dec_se[d]/grad_dec_px[d])/(grad_dec_se.sum()/grad_dec_px.sum()):6.2f}x")

print("\n== SE by region ==")
tp = N * H * W
print(" region         px%     SE%    relSE")
for k in REG:
    print(f" {k:13s} {100*px_sum[k]/tp:6.2f}  {100*se_sum[k]/tot_se:6.2f}"
          f"  {(se_sum[k]/px_sum[k])/(tot_se/tp):6.2f}x")
json.dump(dict(cum=cum.tolist(), qs=qs,
               grad_dec_se=grad_dec_se.tolist(), grad_dec_px=grad_dec_px.tolist(),
               se_sum=se_sum, px_sum=px_sum, tot_se=float(tot_se), tp=int(tp)),
          open(OUT + "/r2.json", "w"))

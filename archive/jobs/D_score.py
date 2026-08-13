"""Where does the SCORE loss live (LPIPS/SSIM spatially), and do diagnosis-derived
combination rules (median / trimmed / var-gated) beat the pixel-mean?"""
import os, sys, json
import numpy as np, cv2, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(8)
dev = "cuda"
SCENE = "HCM0181"; FLD = np.load(f"/mnt/d/avv/fields/{SCENE}.npy")
st = stems(); gd, gm = gt_map(SCENE)
import lpips as L
lpm = L.LPIPS(net='vgg', spatial=True, verbose=False).to(dev).eval()
lps = L.LPIPS(net='vgg', verbose=False).to(dev).eval()
GQ = np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/B_out.npz")['GQ']

MASKS = ["flat(<p50)", "low(p50-80)", "edge(p80-95)", "strong(p95-99)", "vstrong(>p99)",
         "sky", "near-strong-edge(<=3px)", "far-from-edge(>12px)", "ensvar-top10%", "ensvar-bot50%"]
NM = len(MASKS)
a_lp = np.zeros(NM); a_ss = np.zeros(NM); a_se = np.zeros(NM); a_n = np.zeros(NM)
T_lp = T_ss = T_se = 0.0; T_n = 0

# combination rules
COMB = ["mean", "median", "trim1", "vargate_med"]
acc = {c: [0.0, 0.0, 0.0] for c in COMB}

# HP gain analysis: optimal unsharp gain per ensvar decile
NHV = 10
hp_gg = np.zeros(NHV); hp_kk = np.zeros(NHV); hp_gk = np.zeros(NHV); hp_n = np.zeros(NHV)
vq = None

for i, s in enumerate(st):
    G8 = mlib.load_u8(os.path.join(gd, gm[s]))
    raw = [mlib.load_u8(os.path.join(ROOT, m, "test_poses_renders_png", s + ".png")) for m in MEM]
    Mst = np.stack([r.astype(np.float32) for r in raw])              # 4,H,W,3 (0..255)
    mean8 = np.clip(np.round(Mst.mean(0)), 0, 255).astype(np.uint8)
    med8 = np.clip(np.round(np.median(Mst, 0)), 0, 255).astype(np.uint8)
    srt = np.sort(Mst, 0)
    trim8 = np.clip(np.round(srt[1:3].mean(0)), 0, 255).astype(np.uint8)   # == median for k=4
    ensvar = Mst.var(0).mean(-1) / (255.0 ** 2)
    if vq is None:
        vq = np.quantile(ensvar, np.linspace(0, 1, NHV + 1))[1:-1]
    hi = ensvar > np.quantile(ensvar, 0.90)
    vg8 = np.where(hi[..., None], med8, mean8).astype(np.uint8)

    variants = {"mean": mean8, "median": med8, "trim1": trim8, "vargate_med": vg8}
    tg = mlib.to_t(G8, dev)
    for cname, arr in variants.items():
        y = mlib.to_t(apply_field(arr, FLD), dev)
        acc[cname][0] += mlib.psnr(y, tg); acc[cname][1] += float(mlib.ssim(y, tg))
        with torch.no_grad(): acc[cname][2] += float(lps(y * 2 - 1, tg * 2 - 1).item())

    # ---- spatial score decomposition on the production variant (mean+field) ----
    K8 = apply_field(mean8, FLD)
    ty = mlib.to_t(K8, dev)
    with torch.no_grad():
        lmap = lpm(ty * 2 - 1, tg * 2 - 1)[0, 0].cpu().numpy()
    smap = 1.0 - mlib.ssim_map(ty, tg).mean(1)[0].cpu().numpy()
    if lmap.shape != smap.shape:
        lmap = cv2.resize(lmap, (smap.shape[1], smap.shape[0]), interpolation=cv2.INTER_LINEAR)
    G = G8.astype(np.float32) / 255.; K = K8.astype(np.float32) / 255.
    se = ((G - K) ** 2).mean(-1)
    gg = cv2.cvtColor(G8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    gx = cv2.Sobel(gg, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gg, cv2.CV_32F, 0, 1, 3)
    gmag = np.sqrt(gx * gx + gy * gy)
    mu = cv2.blur(gg, (9, 9)); sd = np.sqrt(np.maximum(cv2.blur(gg * gg, (9, 9)) - mu * mu, 0))
    sky = (gg > 0.55) & (sd < 0.012) & (G[..., 2] >= G[..., 0] - 0.02)
    dist = cv2.distanceTransform(1 - (gmag > GQ[2]).astype(np.uint8), cv2.DIST_L2, 3)
    evw = cv2.resize(ensvar, (gg.shape[1], gg.shape[0]))
    ml = [gmag < GQ[0], (gmag >= GQ[0]) & (gmag < GQ[1]), (gmag >= GQ[1]) & (gmag < GQ[2]),
          (gmag >= GQ[2]) & (gmag < GQ[3]), gmag >= GQ[3], sky, dist <= 3, dist > 12,
          evw > np.quantile(evw, 0.90), evw < np.quantile(evw, 0.50)]
    for j, mk in enumerate(ml):
        a_lp[j] += lmap[mk].sum(); a_ss[j] += smap[mk].sum(); a_se[j] += se[mk].sum(); a_n[j] += mk.sum()
    T_lp += lmap.sum(); T_ss += smap.sum(); T_se += se.sum(); T_n += se.size

    # ---- HP gain per ensvar decile (is our HF too weak where members AGREE?) ----
    gl = gg; kl = cv2.cvtColor(K8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    hg = gl - cv2.GaussianBlur(gl, (0, 0), 1.2)
    hk = kl - cv2.GaussianBlur(kl, (0, 0), 1.2)
    bi = np.clip(np.digitize(ensvar, vq), 0, NHV - 1)
    hp_gg += np.bincount(bi.ravel(), weights=(hg * hg).ravel(), minlength=NHV)
    hp_kk += np.bincount(bi.ravel(), weights=(hk * hk).ravel(), minlength=NHV)
    hp_gk += np.bincount(bi.ravel(), weights=(hg * hk).ravel(), minlength=NHV)
    hp_n += np.bincount(bi.ravel(), minlength=NHV)
    if i % 10 == 0: print("img", i, flush=True)

n = len(st)
print("\n=== combination rules (all +field, HCM0181, 60 real test views) ===")
base = None
for c in COMB:
    P, S, Lv = acc[c][0] / n, acc[c][1] / n, acc[c][2] / n
    sc = mlib.score(P, S, Lv)
    if base is None: base = sc
    print(f"  {c:14s} PSNR {P:.4f} SSIM {S:.5f} LPIPS {Lv:.5f}  score {sc:.4f}  ({sc-base:+.4f})")

print("\n=== SCORE-SPACE spatial decomposition (production = k4 mean + field) ===")
print(f"  {'mask':26s} {'%px':>6s} {'%LPIPS':>8s} {'%(1-SSIM)':>10s} {'%SE':>7s} "
      f"{'LPIPSdens':>10s} {'1-SSIMdens':>11s}")
for j, nm in enumerate(MASKS):
    print(f"  {nm:26s} {100*a_n[j]/T_n:6.2f} {100*a_lp[j]/T_lp:8.2f} {100*a_ss[j]/T_ss:10.2f} "
          f"{100*a_se[j]/T_se:7.2f} {a_lp[j]/a_n[j]/(T_lp/T_n):10.3f} {a_ss[j]/a_n[j]/(T_ss/T_n):11.3f}")

print("\n=== HP(sigma1.2) optimal unsharp gain per ensemble-variance decile ===")
print("  dec  meanVar     ampratio |HPk|/|HPg|  coher   a*=<g,k>/<k,k>  MSEred_if_a*  %HPresid")
tot_hpres = (hp_gg + hp_kk - 2 * hp_gk).sum()
for k in range(NHV):
    gg_, kk_, gk_ = hp_gg[k], hp_kk[k], hp_gk[k]
    res = gg_ + kk_ - 2 * gk_
    astar = gk_ / kk_
    red = res - (gg_ - gk_ ** 2 / kk_)
    print(f"   {k+1:2d}  {hp_n[k]:.0f}px  {np.sqrt(kk_/gg_):9.4f}   {gk_/np.sqrt(gg_*kk_):7.4f}"
          f"   {astar:8.4f}       {100*red/res:8.2f}%     {100*res/tot_hpres:6.2f}")
json.dump({"comb": {c: acc[c] for c in COMB}, "n": n}, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/D.json", "w"))

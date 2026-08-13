"""Is 'LPIPS lives in flat regions' actionable?
1) per-VGG-layer LPIPS + per-layer spatial density by GT-gradient mask
2) direct intervention: denoise gated on the RENDER's gradient (test-time available)"""
import os, sys
import numpy as np, cv2, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
import lpips as L
cv2.setNumThreads(6)
dev = "cuda"
SCENE = "HCM0181"; FLD = np.load(f"/mnt/d/avv/fields/{SCENE}.npy")
st = stems(); gd, gm = gt_map(SCENE)
GQ = np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/B_out.npz")['GQ']
lp = mlib.LP(dev)
mS = L.LPIPS(net='vgg', spatial=True, verbose=False).to(dev).eval()

MASKS = ["flat(<p50)", "low(p50-80)", "edge(p80-95)", "strong(>p95)"]
NL = 5; NM = 4
lay_tot = np.zeros(NL)
lay_mask = np.zeros((NL, NM)); mask_n = np.zeros(NM)

# interventions gated on RENDER gradient (available at test time)
INT = ["base", "bil_d5_s10", "bil_d5_s20", "gauss0.5_flat", "gauss0.8_flat", "med3_flat"]
acc = {v: [0.0, 0.0, 0.0] for v in INT}


@torch.no_grad()
def layer_spatial(a, b):
    i0, i1 = mS.scaling_layer(a * 2 - 1), mS.scaling_layer(b * 2 - 1)
    f0, f1 = mS.net.forward(i0), mS.net.forward(i1)
    outs = []
    for k in range(mS.L):
        d = (L.normalize_tensor(f0[k]) - L.normalize_tensor(f1[k])) ** 2
        v = mS.lins[k](d)
        v = torch.nn.functional.interpolate(v, size=a.shape[-2:], mode='bilinear', align_corners=False)
        outs.append(v[0, 0].cpu().numpy())
    return outs


for i, s in enumerate(st):
    G8 = mlib.load_u8(os.path.join(gd, gm[s]))
    K8 = apply_field(mlib.load_u8(os.path.join(K4, s + ".png")), FLD)
    tg = mlib.to_t(G8, dev); ty = mlib.to_t(K8, dev)
    Ls = layer_spatial(ty, tg)
    gray = cv2.cvtColor(G8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    gmag = np.sqrt(gx * gx + gy * gy)
    ml = [gmag < GQ[0], (gmag >= GQ[0]) & (gmag < GQ[1]),
          (gmag >= GQ[1]) & (gmag < GQ[2]), gmag >= GQ[2]]
    for k in range(NL):
        lay_tot[k] += Ls[k].mean()
        for j, mk in enumerate(ml):
            lay_mask[k, j] += Ls[k][mk].sum()
    for j, mk in enumerate(ml):
        mask_n[j] += mk.sum()

    # ---- interventions ----
    rgray = cv2.cvtColor(K8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    rx = cv2.Sobel(rgray, cv2.CV_32F, 1, 0, 3); ry = cv2.Sobel(rgray, cv2.CV_32F, 0, 1, 3)
    rmag = np.sqrt(rx * rx + ry * ry)
    rthr = np.quantile(rmag, 0.50)
    flat = (rmag < rthr)[..., None]
    outs = {"base": K8}
    outs["bil_d5_s10"] = cv2.bilateralFilter(K8, 5, 10, 5)
    outs["bil_d5_s20"] = cv2.bilateralFilter(K8, 5, 20, 5)
    for sg, nm in [(0.5, "gauss0.5_flat"), (0.8, "gauss0.8_flat")]:
        b = cv2.GaussianBlur(K8, (0, 0), sg)
        outs[nm] = np.where(flat, b, K8).astype(np.uint8)
    outs["med3_flat"] = np.where(flat, cv2.medianBlur(K8, 3), K8).astype(np.uint8)
    for v, a in outs.items():
        y = mlib.to_t(a, dev)
        acc[v][0] += mlib.psnr(y, tg); acc[v][1] += float(mlib.ssim(y, tg)); acc[v][2] += lp(y, tg)
    if i % 15 == 0: print("img", i, flush=True)

n = len(st)
print("\n=== LPIPS per VGG layer (production k4+field, HCM0181) ===")
NAMES = ["relu1_2(full)", "relu2_2(1/2)", "relu3_3(1/4)", "relu4_3(1/8)", "relu5_3(1/16)"]
T = lay_tot.sum()
print(f"  {'layer':16s} {'LPIPS':>8s} {'%oftotal':>9s} | density by GT-gradient mask")
print(f"  {'':16s} {'':8s} {'':9s} | " + " ".join(f"{m:>13s}" for m in MASKS))
for k in range(NL):
    dens = [lay_mask[k, j] / mask_n[j] / (lay_mask[k].sum() / mask_n.sum()) for j in range(NM)]
    print(f"  {NAMES[k]:16s} {lay_tot[k]/n:8.5f} {100*lay_tot[k]/T:9.2f} | "
          + " ".join(f"{d:13.3f}" for d in dens))

print("\n=== interventions gated on RENDER gradient (test-time available) ===")
b0 = None
for v in INT:
    P, S, Lv = [x / n for x in acc[v]]
    sc = mlib.score(P, S, Lv)
    if b0 is None: b0 = sc
    print(f"  {v:16s} PSNR {P:8.4f} SSIM {S:.5f} LPIPS {Lv:.5f} score {sc:9.4f} ({sc-b0:+.4f})")

"""LENS: what does LPIPS-VGG actually charge us for, on the production harness?

Oracle SUBSTITUTION decomposition (diagnostic only, uses public_set test GT = legal surface):
  - spatial-frequency bands (5 octave bands + lowpass residual)
  - luma vs chroma
plus per-VGG-layer split, spatial density by distance-to-border and by GT gradient.
Two legal (non-oracle) arms carried along so the report is not oracle-only.
"""
import os, sys, json
import numpy as np, cv2, torch
sys.path.insert(0, "/mnt/d/avv/metric_probe")
import mlib
import lpips as L

cv2.setNumThreads(6)
dev = "cuda"
GTD = "/mnt/d/avv/data/phase1/public_set/{s}/test/images"
SCENES = [("HCM0181", "/mnt/d/avv/prodharness/k4/png"),
          ("HCM0193", "/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png"),
          ("HCM0204", "/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png"),
          ("hcm0031", "/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png"),
          ("hcm0034", "/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png")]
NIMG = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lp = L.LPIPS(net='vgg', verbose=False).to(dev).eval()
mS = L.LPIPS(net='vgg', spatial=True, verbose=False).to(dev).eval()
SIG = [0.8, 1.6, 3.2, 6.4, 12.8]
NB = len(SIG) + 1  # 5 bands + lowpass residual
NL = 5
LNAMES = ["relu1_2(1/1)", "relu2_2(1/2)", "relu3_3(1/4)", "relu4_3(1/8)", "relu5_3(1/16)"]


def apply_field(img8, field):
    H, W, _ = img8.shape
    fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(img8.astype(np.float32) / 255.0,
                    (xx + fu[..., 0]).astype(np.float32),
                    (yy + fu[..., 1]).astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def pyr(img):
    """img float HxWx3 -> list of 5 bands + lowpass residual (sums exactly to img)."""
    outs, cur = [], img
    for s in SIG:
        lo = cv2.GaussianBlur(img, (0, 0), s)
        outs.append(cur - lo)
        cur = lo
    outs.append(cur)
    return outs


@torch.no_grad()
def layer_spatial(a, b):
    i0, i1 = mS.scaling_layer(a * 2 - 1), mS.scaling_layer(b * 2 - 1)
    f0, f1 = mS.net.forward(i0), mS.net.forward(i1)
    outs = []
    for k in range(mS.L):
        d = (L.normalize_tensor(f0[k]) - L.normalize_tensor(f1[k])) ** 2
        v = mS.lins[k](d)
        v = torch.nn.functional.interpolate(v, size=a.shape[-2:], mode='bilinear',
                                            align_corners=False)
        outs.append(v[0, 0].float().cpu().numpy())
    return outs


ARMS = ["base"] + [f"bandGT_{i}" for i in range(NB)] + ["lumaGT", "chromaGT",
                                                        "bil_d5_s10", "gauss0.5_flat"]
res = {}
for scene, rdir in SCENES:
    FLD = np.load(f"/mnt/d/avv/fields/{scene}.npy")
    gd = GTD.format(s=scene)
    gmap = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(gd))}
    stems = sorted(s for s in gmap if os.path.exists(os.path.join(rdir, s + ".png")))
    stems = stems[::max(1, len(stems) // NIMG)][:NIMG]
    acc = {a: np.zeros(3) for a in ARMS}
    lay = np.zeros(NL)
    RINGS = [8, 32, 96]
    ring_sum = np.zeros(len(RINGS) + 1); ring_n = np.zeros(len(RINGS) + 1)
    gq_sum = np.zeros(4); gq_n = np.zeros(4)
    lay_ring = np.zeros((NL, len(RINGS) + 1))
    for i, st in enumerate(stems):
        G8 = mlib.load_u8(os.path.join(gd, gmap[st]))
        R8 = apply_field(mlib.load_u8(os.path.join(rdir, st + ".png")), FLD)
        Gf = G8.astype(np.float32) / 255.
        Rf = R8.astype(np.float32) / 255.
        tg = torch.from_numpy(Gf).permute(2, 0, 1)[None].to(dev)
        variants = {"base": Rf}
        pR, pG = pyr(Rf), pyr(Gf)
        for j in range(NB):
            variants[f"bandGT_{j}"] = np.clip(Rf - pR[j] + pG[j], 0, 1)
        Ry = cv2.cvtColor(Rf, cv2.COLOR_RGB2YCrCb); Gy = cv2.cvtColor(Gf, cv2.COLOR_RGB2YCrCb)
        v = Ry.copy(); v[..., 0] = Gy[..., 0]
        variants["lumaGT"] = np.clip(cv2.cvtColor(v, cv2.COLOR_YCrCb2RGB), 0, 1)
        v = Ry.copy(); v[..., 1:] = Gy[..., 1:]
        variants["chromaGT"] = np.clip(cv2.cvtColor(v, cv2.COLOR_YCrCb2RGB), 0, 1)
        variants["bil_d5_s10"] = cv2.bilateralFilter(R8, 5, 10, 5).astype(np.float32) / 255.
        rg = cv2.cvtColor(R8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        rmag = np.hypot(cv2.Sobel(rg, cv2.CV_32F, 1, 0, 3), cv2.Sobel(rg, cv2.CV_32F, 0, 1, 3))
        flat = (rmag < np.quantile(rmag, 0.5))[..., None]
        variants["gauss0.5_flat"] = np.where(flat, cv2.GaussianBlur(Rf, (0, 0), 0.5), Rf)
        for a, arr in variants.items():
            t = torch.from_numpy(np.ascontiguousarray(np.clip(arr, 0, 1))).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                acc[a] += [mlib.psnr(t, tg), float(mlib.ssim(t, tg)),
                           float(lp(t * 2 - 1, tg * 2 - 1).item())]
        # spatial
        tb = torch.from_numpy(np.ascontiguousarray(Rf)).permute(2, 0, 1)[None].to(dev)
        Ls = layer_spatial(tb, tg)
        H, W = Rf.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        d2b = np.minimum(np.minimum(yy, H - 1 - yy), np.minimum(xx, W - 1 - xx))
        rmasks = [d2b < RINGS[0]] + [(d2b >= RINGS[k]) & (d2b < RINGS[k + 1])
                                     for k in range(len(RINGS) - 1)] + [d2b >= RINGS[-1]]
        gg = cv2.cvtColor(G8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        gmag = np.hypot(cv2.Sobel(gg, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gg, cv2.CV_32F, 0, 1, 3))
        gq = np.quantile(gmag, [0.5, 0.8, 0.95])
        gmasks = [gmag < gq[0], (gmag >= gq[0]) & (gmag < gq[1]),
                  (gmag >= gq[1]) & (gmag < gq[2]), gmag >= gq[2]]
        tot = sum(x for x in Ls)
        for k in range(NL):
            lay[k] += Ls[k].mean()
            for j, m in enumerate(rmasks):
                lay_ring[k, j] += Ls[k][m].sum()
        for j, m in enumerate(rmasks):
            ring_sum[j] += tot[m].sum(); ring_n[j] += m.sum()
        for j, m in enumerate(gmasks):
            gq_sum[j] += tot[m].sum(); gq_n[j] += m.sum()
    n = len(stems)
    res[scene] = dict(n=n,
                      arms={a: (acc[a] / n).tolist() for a in ARMS},
                      layers=(lay / n).tolist(),
                      ring_density=(ring_sum / np.maximum(ring_n, 1) /
                                    (ring_sum.sum() / ring_n.sum())).tolist(),
                      ring_share=(ring_sum / ring_sum.sum()).tolist(),
                      ring_pxshare=(ring_n / ring_n.sum()).tolist(),
                      grad_density=(gq_sum / np.maximum(gq_n, 1) /
                                    (gq_sum.sum() / gq_n.sum())).tolist(),
                      grad_share=(gq_sum / gq_sum.sum()).tolist(),
                      lay_ring_density=(lay_ring / np.maximum(ring_n, 1)[None, :] /
                                        (lay_ring.sum(1) / ring_n.sum())[:, None]).tolist())
    b = res[scene]['arms']['base']
    print(f"\n### {scene}  n={n}  base PSNR {b[0]:.3f} SSIM {b[1]:.4f} LPIPS {b[2]:.5f}", flush=True)
    for a in ARMS[1:]:
        v = res[scene]['arms'][a]
        ds = mlib.score(*v) - mlib.score(*b)
        rec = (b[2] - v[2]) / b[2] * 100
        print(f"  {a:14s} LPIPS {v[2]:.5f} ({rec:+6.2f}% of base)  PSNR {v[0]:7.3f} "
              f"SSIM {v[1]:.4f}  dScore {ds:+8.4f}", flush=True)
    print("  layers %:", [f"{100*x/sum(res[scene]['layers']):.1f}" for x in res[scene]['layers']], flush=True)
    print("  ring density (<8,8-32,32-96,>=96 px from border):",
          [f"{x:.3f}" for x in res[scene]['ring_density']],
          " share:", [f"{100*x:.1f}%" for x in res[scene]['ring_share']],
          " px:", [f"{100*x:.1f}%" for x in res[scene]['ring_pxshare']], flush=True)
    print("  grad density (flat,low,edge,strong):", [f"{x:.3f}" for x in res[scene]['grad_density']],
          " share:", [f"{100*x:.1f}%" for x in res[scene]['grad_share']], flush=True)

json.dump(res, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/LENS_lpips.json", "w"), indent=1)

# ---- pooled ----
print("\n\n================ POOLED OVER 5 PUBLIC TOWERS ================")
base = np.mean([res[s]['arms']['base'] for s, _ in SCENES], 0)
print(f"base  PSNR {base[0]:.4f} SSIM {base[1]:.5f} LPIPS {base[2]:.5f} "
      f"score {mlib.score(*base):.4f}")
for a in ARMS[1:]:
    v = np.mean([res[s]['arms'][a] for s, _ in SCENES], 0)
    npos = sum(1 for s, _ in SCENES if res[s]['arms'][a][2] < res[s]['arms']['base'][2])
    print(f"  {a:14s} LPIPS {v[2]:.5f}  recovery {(base[2]-v[2])/base[2]*100:+6.2f}%  "
          f"dScore {mlib.score(*v)-mlib.score(*base):+8.4f}  ({npos}/5 LPIPS-positive)")
lay = np.mean([res[s]['layers'] for s, _ in SCENES], 0)
print("\nper-layer LPIPS:", " ".join(f"{LNAMES[k]}={lay[k]:.5f}({100*lay[k]/lay.sum():.1f}%)"
                                    for k in range(NL)))
rd = np.mean([res[s]['ring_density'] for s, _ in SCENES], 0)
rs = np.mean([res[s]['ring_share'] for s, _ in SCENES], 0)
rp = np.mean([res[s]['ring_pxshare'] for s, _ in SCENES], 0)
print("border rings <8 / 8-32 / 32-96 / >=96 px:")
print("   density", [f"{x:.3f}" for x in rd], " LPIPS share", [f"{100*x:.2f}%" for x in rs],
      " px share", [f"{100*x:.2f}%" for x in rp])
lrd = np.mean([res[s]['lay_ring_density'] for s, _ in SCENES], 0)
for k in range(NL):
    print(f"   {LNAMES[k]:14s} ring density {[f'{x:.3f}' for x in lrd[k]]}")
gd_ = np.mean([res[s]['grad_density'] for s, _ in SCENES], 0)
gs_ = np.mean([res[s]['grad_share'] for s, _ in SCENES], 0)
print("grad quartiles flat/low/edge/strong: density", [f"{x:.3f}" for x in gd_],
      " share", [f"{100*x:.1f}%" for x in gs_])
print("DONE")

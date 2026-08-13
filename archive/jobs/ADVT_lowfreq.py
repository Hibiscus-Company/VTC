"""ADVERSARIAL TRANSFER TEST for the bandGT_5 (lowpass sigma>12.8) claim.

Question: the ORACLE pastes each image's OWN GT lowpass. Any realizable operator must
estimate that lowpass residual WITHOUT that image's GT. Decompose:
  D_i = blur(G_i,12.8) - blur(R_i,12.8)     (the thing the oracle adds)
  (a) how much of sum||D_i||^2 is the VIEW-CONSISTENT mean field  mean_i D_i ?
  (b) how much is a per-image GLOBAL 3-number colour offset (still needs GT, upper bound
      for any global photometric model)?
  (c) LOVO: apply mean_{j!=i} D_j to image i  -> honest score of the legally-estimable part.
      (LOVO here is still GENEROUS: it uses other TEST GTs. A train-fit field can only be worse.)
Arms scored with the real competition metric.
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
SIGMA = 12.8

lp = L.LPIPS(net='vgg', verbose=False).to(dev).eval()


def apply_field(img8, field):
    H, W, _ = img8.shape
    fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(img8.astype(np.float32) / 255.0,
                    (xx + fu[..., 0]).astype(np.float32),
                    (yy + fu[..., 1]).astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def sc(arr, tg):
    t = torch.from_numpy(np.ascontiguousarray(np.clip(arr, 0, 1))).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        return np.array([mlib.psnr(t, tg), float(mlib.ssim(t, tg)), float(lp(t * 2 - 1, tg * 2 - 1).item())])


ARMS = ["base", "oracle_lowpass", "LOVO_meanfield", "oracle_globalRGB", "LOVO_globalRGB"]
res = {}
for scene, rdir in SCENES:
    FLD = np.load(f"/mnt/d/avv/fields/{scene}.npy")
    gd = GTD.format(s=scene)
    gmap = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(gd))}
    stems = sorted(s for s in gmap if os.path.exists(os.path.join(rdir, s + ".png")))
    stems = stems[::max(1, len(stems) // NIMG)][:NIMG]
    Rs, Gs, Ds = [], [], []
    for st in stems:
        G8 = mlib.load_u8(os.path.join(gd, gmap[st]))
        R8 = apply_field(mlib.load_u8(os.path.join(rdir, st + ".png")), FLD)
        Gf = G8.astype(np.float32) / 255.
        Rf = R8.astype(np.float32) / 255.
        D = cv2.GaussianBlur(Gf, (0, 0), SIGMA) - cv2.GaussianBlur(Rf, (0, 0), SIGMA)
        Rs.append(Rf); Gs.append(Gf); Ds.append(D)
    n = len(stems)
    Dst = np.stack(Ds)                      # n,H,W,3
    tot = float((Dst ** 2).mean())
    mean_field = Dst.mean(0)
    f_field = float((mean_field ** 2).mean()) / tot          # view-consistent share
    gmeans = Dst.mean(axis=(1, 2), keepdims=True)            # n,1,1,3 per-image global RGB offset
    f_glob = float((gmeans ** 2).mean()) / tot
    gm_mean = gmeans.mean(0)
    f_globconst = float((gm_mean ** 2).mean()) / tot
    acc = {a: np.zeros(3) for a in ARMS}
    for i in range(n):
        tg = torch.from_numpy(Gs[i]).permute(2, 0, 1)[None].to(dev)
        lo_mean = (Dst.sum(0) - Dst[i]) / (n - 1)             # LOVO mean field
        gl_mean = (gmeans.sum(0) - gmeans[i]) / (n - 1)       # LOVO global RGB
        v = {"base": Rs[i],
             "oracle_lowpass": Rs[i] + Ds[i],
             "LOVO_meanfield": Rs[i] + lo_mean,
             "oracle_globalRGB": Rs[i] + gmeans[i],
             "LOVO_globalRGB": Rs[i] + gl_mean}
        for a in ARMS:
            acc[a] += sc(v[a], tg)
    res[scene] = dict(n=n, arms={a: (acc[a] / n).tolist() for a in ARMS},
                      f_field=f_field, f_glob=f_glob, f_globconst=f_globconst,
                      rms_D=float(np.sqrt(tot)) * 255)
    b = np.array(res[scene]['arms']['base'])
    print(f"\n### {scene} n={n} rmsD {res[scene]['rms_D']:.3f}/255  "
          f"view-consistent share {100*f_field:.1f}%  per-img-globalRGB share {100*f_glob:.1f}%  "
          f"const-globalRGB share {100*f_globconst:.1f}%", flush=True)
    for a in ARMS[1:]:
        v = np.array(res[scene]['arms'][a])
        print(f"  {a:18s} PSNR {v[0]:7.3f} SSIM {v[1]:.4f} LPIPS {v[2]:.5f} "
              f"dScore {mlib.score(*v)-mlib.score(*b):+8.4f}", flush=True)

json.dump(res, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADVT_lowfreq.json", "w"), indent=1)
print("\n============ POOLED 5 TOWERS ============")
base = np.mean([res[s]['arms']['base'] for s, _ in SCENES], 0)
print(f"base PSNR {base[0]:.4f} SSIM {base[1]:.5f} LPIPS {base[2]:.5f} score {mlib.score(*base):.4f}")
for a in ARMS[1:]:
    v = np.mean([res[s]['arms'][a] for s, _ in SCENES], 0)
    npos = sum(1 for s, _ in SCENES
               if mlib.score(*res[s]['arms'][a]) > mlib.score(*res[s]['arms']['base']))
    d = mlib.score(*v) - mlib.score(*base)
    print(f"  {a:18s} PSNR {v[0]:7.4f} SSIM {v[1]:.5f} LPIPS {v[2]:.5f} dScore {d:+8.4f} "
          f"({npos}/5 score-positive)   LBequiv(5/7) {d*5/7:+.4f}")
print("view-consistent share mean %.1f%%  per-img-globalRGB %.1f%%  const-globalRGB %.1f%%" % (
    100*np.mean([res[s]['f_field'] for s, _ in SCENES]),
    100*np.mean([res[s]['f_glob'] for s, _ in SCENES]),
    100*np.mean([res[s]['f_globconst'] for s, _ in SCENES])))
print("DONE")

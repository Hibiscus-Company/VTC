"""ADVERSARIAL: is the sigma>12.8px lowpass residual REACHABLE, or is it per-image content error?

Arms (same 5 towers, same stems, same lens field, same metrics as LENS_lpips.py):
  base        shipped render + lens field
  lpGT_full   bandGT_5 exactly (oracle, must reproduce +0.4553)
  lpGT_ds8    own residual stored at ds=8 -> representation-loss control
  dcGT        per-image per-channel SCALAR offset oracle  (global DC = floor of any global model)
  affGT       per-image per-channel gain+bias LS oracle   (= "global affine colour" on THIS pool)
  looMean     LEAVE-ONE-OUT scene-mean residual field (ds=8): the best a FIXED spatially-varying
              per-scene field can do -- this is the entire actionable content of the claim
  looMeanNoDC LOO mean field with each image's DC removed (shared SHAPE only)
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
SIG = 12.8
DS = 8

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


ARMS = ["base", "lpGT_full", "lpGT_ds8", "dcGT", "affGT", "looMean", "looMeanNoDC"]
res, edec = {}, {}
for scene, rdir in SCENES:
    FLD = np.load(f"/mnt/d/avv/fields/{scene}.npy")
    gd = GTD.format(s=scene)
    gmap = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(gd))}
    stems = sorted(s for s in gmap if os.path.exists(os.path.join(rdir, s + ".png")))
    stems = stems[::max(1, len(stems) // NIMG)][:NIMG]
    # ---- pass 1: residual fields ----
    Ds, cache = [], []
    for st in stems:
        G8 = mlib.load_u8(os.path.join(gd, gmap[st]))
        R8 = apply_field(mlib.load_u8(os.path.join(rdir, st + ".png")), FLD)
        Gf = G8.astype(np.float32) / 255.
        Rf = R8.astype(np.float32) / 255.
        LG = cv2.GaussianBlur(Gf, (0, 0), SIG)
        LR = cv2.GaussianBlur(Rf, (0, 0), SIG)
        D = LG - LR
        H, W = D.shape[:2]
        Ds.append(cv2.resize(D, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
        cache.append((Rf, Gf, D))
    Ds = np.stack(Ds)                      # [N, h, w, 3]
    N = len(stems)
    Dbar = Ds.mean(0)
    dc = Ds.mean((1, 2), keepdims=True)    # [N,1,1,3] per-image per-channel DC
    Et = float((Ds ** 2).mean())
    Edc = float((dc ** 2).mean())
    Ecom = float((Dbar ** 2).mean())
    Espe = float(((Ds - Dbar) ** 2).mean())
    EcomNoDC = float(((Ds - dc).mean(0) ** 2).mean())
    edec[scene] = dict(Etot=Et, E_perimage_DC=Edc, E_scene_common=Ecom,
                       E_image_specific=Espe, E_common_shape_noDC=EcomNoDC, N=N)
    print(f"\n### {scene} n={N}  residual energy (lowpass sigma={SIG}) ---"
          f" tot {Et:.3e} | per-image DC {100*Edc/Et:5.1f}% | scene-common field {100*Ecom/Et:5.1f}%"
          f" (of which shape-only {100*EcomNoDC/Et:5.1f}%) | image-specific {100*Espe/Et:5.1f}%",
          flush=True)
    # ---- pass 2: arms ----
    acc = {a: np.zeros(3) for a in ARMS}
    for i, st in enumerate(stems):
        Rf, Gf, D = cache[i]
        H, W = Rf.shape[:2]
        up = lambda d: cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
        loo = (Ds.sum(0) - Ds[i]) / (N - 1)
        looN = ((Ds - dc).sum(0) - (Ds[i] - dc[i])) / (N - 1)
        v = {}
        v["base"] = Rf
        v["lpGT_full"] = Rf + D
        v["lpGT_ds8"] = Rf + up(Ds[i])
        v["dcGT"] = Rf + D.mean((0, 1))[None, None, :]
        # per-channel gain+bias LS fit of render -> GT (global affine colour oracle)
        aff = Rf.copy()
        for c in range(3):
            x = Rf[..., c].ravel(); y = Gf[..., c].ravel()
            xm, ym = x.mean(), y.mean()
            g = float(((x - xm) * (y - ym)).sum() / max(((x - xm) ** 2).sum(), 1e-9))
            aff[..., c] = g * Rf[..., c] + (ym - g * xm)
        v["affGT"] = aff
        v["looMean"] = Rf + up(loo)
        v["looMeanNoDC"] = Rf + up(looN)
        tg = torch.from_numpy(np.ascontiguousarray(Gf)).permute(2, 0, 1)[None].to(dev)
        for a, arr in v.items():
            t = torch.from_numpy(np.ascontiguousarray(np.clip(arr, 0, 1))).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                acc[a] += [mlib.psnr(t, tg), float(mlib.ssim(t, tg)),
                           float(lp(t * 2 - 1, tg * 2 - 1).item())]
    res[scene] = {a: (acc[a] / N).tolist() for a in ARMS}
    b = res[scene]["base"]
    for a in ARMS[1:]:
        vv = res[scene][a]
        print(f"  {a:12s} PSNR {vv[0]:7.3f} SSIM {vv[1]:.4f} LPIPS {vv[2]:.5f} "
              f"dScore {mlib.score(*vv)-mlib.score(*b):+8.4f}", flush=True)

json.dump(dict(arms=res, edec=edec), open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/REF_lowfreq.json", "w"), indent=1)
print("\n============ POOLED OVER 5 TOWERS ============")
b = np.mean([res[s][ "base"] for s, _ in SCENES], 0)
print(f"base PSNR {b[0]:.4f} SSIM {b[1]:.5f} LPIPS {b[2]:.5f} score {mlib.score(*b):.4f}")
for a in ARMS[1:]:
    vv = np.mean([res[s][a] for s, _ in SCENES], 0)
    npos = sum(1 for s, _ in SCENES if mlib.score(*res[s][a]) > mlib.score(*res[s]["base"]))
    print(f"  {a:12s} PSNR {vv[0]:7.4f} SSIM {vv[1]:.5f} LPIPS {vv[2]:.5f}  "
          f"dScore {mlib.score(*vv)-mlib.score(*b):+8.4f}  ({npos}/5 positive)")
E = {k: np.mean([edec[s][k] for s, _ in SCENES]) for k in
     ["Etot", "E_perimage_DC", "E_scene_common", "E_image_specific", "E_common_shape_noDC"]}
print("\nPOOLED residual energy split: per-image DC %.1f%%  scene-common field %.1f%% "
      "(shape-only %.1f%%)  image-specific %.1f%%"
      % (100*E["E_perimage_DC"]/E["Etot"], 100*E["E_scene_common"]/E["Etot"],
         100*E["E_common_shape_noDC"]/E["Etot"], 100*E["E_image_specific"]/E["Etot"]))
print("DONE")

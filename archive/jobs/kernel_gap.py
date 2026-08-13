import os, sys, numpy as np, cv2, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SC = sys.argv[1] if len(sys.argv) > 1 else "HCM0421"
DIR = f"/mnt/d/avv/r31/tower_ens/{SC}/png"
FLD = f"/mnt/d/avv/fields_median_g1_g130/{SC}.npy"
field = np.load(FLD).astype(np.float32)          # [h,w,2] at 1/8
files = sorted(os.listdir(DIR))[:8]

def lanczos_warp(img, fu):
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, (xx + fu[..., 0]).astype(np.float32),
                     (yy + fu[..., 1]).astype(np.float32),
                     cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

def torch_warp(img, fu, mode):
    H, W, _ = img.shape
    t = torch.from_numpy(img).permute(2, 0, 1)[None]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx = (xx + fu[..., 0]); gy = (yy + fu[..., 1])
    gx = 2.0 * gx / (W - 1) - 1.0
    gy = 2.0 * gy / (H - 1) - 1.0
    g = torch.from_numpy(np.stack([gx, gy], -1))[None]
    o = torch.nn.functional.grid_sample(t, g, mode=mode, padding_mode="reflection",
                                        align_corners=True)
    return o[0].permute(1, 2, 0).numpy()

def lap_energy(x):
    g = x.mean(-1)
    return float((cv2.Laplacian(g, cv2.CV_32F, ksize=3) ** 2).mean())

def psnr(a, b):
    return 10 * np.log10(1.0 / max(((a - b) ** 2).mean(), 1e-12))

rows = []
for f in files:
    img = np.asarray(Image.open(os.path.join(DIR, f)).convert("RGB"), np.float32) / 255.
    H, W, _ = img.shape
    fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    e0 = lap_energy(img)
    L = lanczos_warp(img, fu)
    B = torch_warp(img, fu, "bilinear")
    C = torch_warp(img, fu, "bicubic")
    rows.append((lap_energy(L) / e0, lap_energy(B) / e0, lap_energy(C) / e0,
                 psnr(L, B), psnr(L, C), np.abs(fu).mean()))
r = np.array(rows)
m = r.mean(0)
print(f"{SC}  n={len(files)}  {W}x{H}  mean|d| {m[5]:.3f}px")
print(f"  Laplacian-energy retention (1 forward warp):")
print(f"    cv2 LANCZOS4 (production)     {m[0]:.4f}")
print(f"    torch grid_sample bilinear    {m[1]:.4f}   (proposal's kernel)")
print(f"    torch grid_sample bicubic     {m[2]:.4f}")
print(f"  PSNR(lanczos4 vs bilinear) {m[3]:.2f} dB   PSNR(lanczos4 vs bicubic) {m[4]:.2f} dB")

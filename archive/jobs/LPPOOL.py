"""Is the LOW-FREQUENCY residual view-CONSISTENT (=> a pooled photometric field, the twin of the
shipped lens field) or per-view? Leave-one-view-out median pool, scored with the project metric."""
import os, sys
import numpy as np, cv2, torch
sys.path.insert(0, "/mnt/d/avv/metric_probe")
import mlib
import lpips as L
cv2.setNumThreads(6)
dev = os.environ.get("DEV","cuda:1")
GTD = "/mnt/d/avv/data/phase1/public_set/{s}/test/images"
SCENE, SRC, USEFIELD, NIMG = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
SIG = 32
FLD = np.load(f"/mnt/d/avv/fields/{SCENE}.npy") if USEFIELD else None


def apply_field(img8, field, scale=1.30):
    H, W, _ = img8.shape
    fu = cv2.resize(field * scale, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(img8.astype(np.float32) / 255.0, (xx + fu[..., 0]).astype(np.float32),
                    (yy + fu[..., 1]).astype(np.float32), cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


gd = GTD.format(s=SCENE)
gm = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(gd))}
stems = [s for s in (os.path.splitext(f)[0] for f in sorted(os.listdir(SRC))) if s in gm][:NIMG]
lp = L.LPIPS(net='vgg', verbose=False).to(dev).eval()

bases, gts, Rs = [], [], []
for st in stems:
    G8 = mlib.load_u8(os.path.join(gd, gm[st]))
    B8 = mlib.load_u8(os.path.join(SRC, st + ".png"))
    if B8.shape != G8.shape:
        B8 = cv2.resize(B8, (G8.shape[1], G8.shape[0]), interpolation=cv2.INTER_CUBIC)
    if FLD is not None: B8 = apply_field(B8, FLD)
    R = cv2.GaussianBlur(G8.astype(np.float32) - B8.astype(np.float32), (0, 0), SIG,
                         borderType=cv2.BORDER_REFLECT)
    bases.append(B8); gts.append(G8)
    Rs.append(cv2.resize(R, (R.shape[1] // 8, R.shape[0] // 8), interpolation=cv2.INTER_AREA))
Rs = np.stack(Rs)  # [N,h,w,3]
N = len(stems)
Etot = float((Rs ** 2).mean())
M = np.median(Rs, axis=0)
Eres = float(((Rs - M) ** 2).mean())
print(f"\n=== POOLED LOW-FREQ PHOTOMETRIC FIELD  {SCENE}  n={N}  sigma={SIG} ===")
print(f"  per-view LF residual RMS      : {np.sqrt(Etot):.3f} LSB")
print(f"  pooled median field RMS       : {np.sqrt((M**2).mean()):.3f} LSB   (max |M| {np.abs(M).max():.2f})")
print(f"  residual after pooled removal : {np.sqrt(Eres):.3f} LSB")
print(f"  VIEW-CONSISTENT energy fraction: {100*(1-Eres/Etot):.2f}%")

acc = {"base": [0., 0., 0.], "pooledLOO": [0., 0., 0.]}
H, W = bases[0].shape[:2]
with torch.no_grad():
    for i in range(N):
        g_t = mlib.to_t(gts[i], dev)
        Mi = np.median(np.delete(Rs, i, axis=0), axis=0)
        Mi = cv2.resize(Mi, (W, H), interpolation=cv2.INTER_CUBIC)
        out = np.clip(bases[i].astype(np.float32) + Mi, 0, 255).astype(np.uint8)
        for k, a in (("base", bases[i]), ("pooledLOO", out)):
            y = mlib.to_t(a, dev)
            acc[k][0] += mlib.psnr(y, g_t); acc[k][1] += float(mlib.ssim(y, g_t))
            acc[k][2] += float(lp(y * 2 - 1, g_t * 2 - 1))
b = None
for k in ("base", "pooledLOO"):
    P, S, Lv = [x / N for x in acc[k]]
    s0 = mlib.score(P, S, Lv)
    if b is None: b = (s0, Lv)
    print(f"  {k:10s} PSNR {P:8.4f} SSIM {S:.5f} LPIPS {Lv:.5f} score {s0:9.4f} "
          f"({s0-b[0]:+.4f}, dLPIPS {Lv-b[1]:+.5f})")

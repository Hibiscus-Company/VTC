"""ORACLE frequency partition of the shipped-chain residual.
out_LF(s)  = base + Gauss(GT-base, s)      <- upper bound for ANY low-freq-only per-image op
out_HF(s)  = base + (GT-base) - Gauss(...)  <- complement (detail energy only)
Diagnostic only; public_set GT is sanctioned.
"""
import os, sys, json
import numpy as np, cv2, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
sys.path.insert(0, "/mnt/d/avv/metric_probe")
import mlib
import lpips as L
cv2.setNumThreads(6)
dev = "cuda:1"

GTD = "/mnt/d/avv/data/phase1/public_set/{s}/test/images"
SCENE = sys.argv[1] if len(sys.argv) > 1 else "HCM0181"
SRC = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/avv/prodharness/k4/png"
USEFIELD = int(sys.argv[3]) if len(sys.argv) > 3 else 1
NIMG = int(sys.argv[4]) if len(sys.argv) > 4 else 999

FLD = np.load(f"/mnt/d/avv/fields/{SCENE}.npy") if USEFIELD else None


def apply_field(img8, field, scale=1.30):
    H, W, _ = img8.shape
    fu = cv2.resize(field * scale, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(img8.astype(np.float32) / 255.0,
                    (xx + fu[..., 0]).astype(np.float32),
                    (yy + fu[..., 1]).astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


gd = GTD.format(s=SCENE)
gm = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(gd))}
stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(SRC))]
stems = [s for s in stems if s in gm][:NIMG]
print(f"{SCENE}: {len(stems)} paired views, src={SRC} field={USEFIELD}", flush=True)

lp = L.LPIPS(net='vgg', verbose=False).to(dev).eval()
SIG = [2, 4, 8, 16, 32, 64]
ARMS = ["base"] + [f"LF{s}" for s in SIG] + ["HF16", "HF32"]
acc = {a: [0.0, 0.0, 0.0] for a in ARMS}
# residual energy fractions
efrac = np.zeros(len(SIG)); ecnt = 0


@torch.no_grad()
def sc(a8, g_t):
    y = mlib.to_t(a8, dev)
    return mlib.psnr(y, g_t), float(mlib.ssim(y, g_t)), float(lp(y * 2 - 1, g_t * 2 - 1))


for i, st in enumerate(stems):
    G8 = mlib.load_u8(os.path.join(gd, gm[st]))
    B8 = mlib.load_u8(os.path.join(SRC, st + ".png"))
    if B8.shape != G8.shape:
        B8 = cv2.resize(B8, (G8.shape[1], G8.shape[0]), interpolation=cv2.INTER_CUBIC)
    if FLD is not None:
        B8 = apply_field(B8, FLD)
    g_t = mlib.to_t(G8, dev)
    Bf = B8.astype(np.float32)
    D = G8.astype(np.float32) - Bf
    outs = {"base": B8}
    for k, s in enumerate(SIG):
        Ds = cv2.GaussianBlur(D, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        outs[f"LF{s}"] = np.clip(Bf + Ds, 0, 255).astype(np.uint8)
        if s in (16, 32):
            outs[f"HF{s}"] = np.clip(Bf + (D - Ds), 0, 255).astype(np.uint8)
        efrac[k] += float((Ds ** 2).mean() / max((D ** 2).mean(), 1e-9))
    ecnt += 1
    for a in ARMS:
        P, S, Lv = sc(outs[a], g_t)
        acc[a][0] += P; acc[a][1] += S; acc[a][2] += Lv
    if i % 10 == 0:
        print("  img", i, flush=True)

n = len(stems)
print(f"\n=== ORACLE FREQ PARTITION  {SCENE}  n={n} ===")
print(f"  {'arm':8s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s} {'SCORE':>9s} {'dSCORE':>9s} {'dLPIPS':>9s}")
b = None
res = {}
for a in ARMS:
    P, S, Lv = [x / n for x in acc[a]]
    s0 = mlib.score(P, S, Lv)
    if b is None: b = (s0, Lv)
    print(f"  {a:8s} {P:8.4f} {S:8.5f} {Lv:8.5f} {s0:9.4f} {s0-b[0]:+9.4f} {Lv-b[1]:+9.5f}")
    res[a] = dict(psnr=P, ssim=S, lpips=Lv, score=s0)
print("\n  residual ENERGY fraction captured by Gauss(sigma):")
for k, s in enumerate(SIG):
    print(f"    sigma {s:3d}: {100*efrac[k]/ecnt:6.2f}% of residual squared energy")
json.dump(res, open(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/LPFREQ_{SCENE}.json", "w"), indent=1)

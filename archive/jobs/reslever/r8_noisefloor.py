"""R8: how much of the residual is un-renderable GT sensor/JPEG noise?"""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
VIEWS = list(range(0, 60, 3))

# HP filter energy gain for white noise
z = np.random.RandomState(0).randn(512, 512).astype(np.float32)
gain = ((z - cv2.GaussianBlur(z, (0, 0), 1.0)) ** 2).mean()
print("HP(sigma=1) white-noise energy gain = %.4f" % gain)

hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.0)
acc = dict(sky_g=0., sky_p=0., sky_r=0., sky_n=0, flat_g=0., flat_p=0., flat_r=0., flat_n=0)
for i in VIEWS:
    g = G[i].astype(np.float32) / 255.0
    p = apply_field(R[k4i, i].astype(np.float32) / 255.0, field)
    y = (g * np.array([.299, .587, .114], np.float32)).sum(2)
    yp = (p * np.array([.299, .587, .114], np.float32)).sum(2)
    ys = cv2.GaussianBlur(y, (0, 0), 2.0)
    gm = cv2.GaussianBlur(np.sqrt(cv2.Sobel(ys, cv2.CV_32F, 1, 0, 3) ** 2 +
                                  cv2.Sobel(ys, cv2.CV_32F, 0, 1, 3) ** 2), (0, 0), 3.0)
    for tag, q in [("sky", 0.05), ("flat", 0.25)]:
        m = gm < np.quantile(gm, q)
        acc[tag + "_g"] += (hp(y)[m] ** 2).sum()
        acc[tag + "_p"] += (hp(yp)[m] ** 2).sum()
        acc[tag + "_r"] += (hp(yp - y)[m] ** 2).sum()
        acc[tag + "_n"] += m.sum()
print("\nregion    E|HP gt|^2   E|HP pred|^2  E|HP resid|^2   sum(g+p)   -> indep?")
for tag in ["sky", "flat"]:
    n = acc[tag + "_n"]
    a, b, c = acc[tag + "_g"] / n, acc[tag + "_p"] / n, acc[tag + "_r"] / n
    print(f" {tag:6s}  {a:.4e}  {b:.4e}  {c:.4e}   {a+b:.4e}   ratio r/(g+p)={c/(a+b):.3f}")
    print(f"         -> implied sigma_noise (if HP gt is pure noise) = "
          f"{np.sqrt(a/gain)*255:.2f}/255 ; var = {a/gain:.3e}")

sig2 = acc["sky_g"] / acc["sky_n"] / gain
print(f"\nGT noise variance estimate (from flattest 5%%): {sig2:.3e}  "
      f"= {100*sig2/ (10**(-25.923/10)):.1f}%% of our k4 MSE ({10**(-25.923/10):.3e})")
print(f"PSNR ceiling if the ONLY error were unpredictable GT noise: "
      f"{10*np.log10(1/sig2):.2f} dB")

# ---- oracle ceiling: what does a perfect but noise-free renderer score? ----
print("\n== score of denoised-GT vs GT (proxy for a perfect noise-free renderer) ==")
for h in [1.0, 2.0, 3.0, 4.0]:
    P = S = L = 0.0
    rem = 0.0
    for i in VIEWS:
        g = G[i].astype(np.float32) / 255.0
        u = (g * 255).astype(np.uint8)
        d = cv2.fastNlMeansDenoisingColored(u, None, h, h, 7, 21).astype(np.float32) / 255.0
        rem += ((d - g) ** 2).mean()
        a, b, c = score(d, g)
        P += a; S += b; L += c
    n = len(VIEWS)
    print(f" NLM h={h}: removed MSE {rem/n:.3e}  PSNR {P/n:7.4f} SSIM {S/n:.5f} "
          f"LPIPS {L/n:.5f}  comp {comp(P/n,S/n,L/n):8.4f}")

import os, json, glob
import numpy as np
CACHE = "/mnt/d/avv/r42_bonsai78/a1_oracle/cache"


def sc(p, s, l):
    return 100.0 * (0.4 * (1 - l) + 0.3 * s + 0.3 * min(p / 50.0, 1.0))


D = {}
for f in glob.glob(os.path.join(CACHE, "*.json")):
    r = json.load(open(f))
    D.setdefault(r["stem"], {})[r["key"]] = r
stems = sorted(D)
have = sorted(set.intersection(*[set(D[s]) for s in stems]))


def agg(k):
    P = np.mean([D[s][k]["psnr"] for s in stems])
    S = np.mean([D[s][k]["ssim"] for s in stems])
    L = np.mean([D[s][k]["lpips"] for s in stems])
    return P, S, L, sc(P, S, L)


b = agg("base")
print("BASE  PSNR %.4f SSIM %.5f LPIPS %.5f SCORE %.4f\n" % b)
order = ["us_0.6_-1.0", "us_0.6_-0.3", "us_1.6_-0.3", "us_0.6_-0.15", "us_1.6_-0.15", "base",
         "us_0.6_0.15", "us_1.6_0.15", "spec_pc", "spec_sh", "specw_pc", "specw_sh"]
print("%-14s %8s %8s %9s %9s | %8s %8s %8s %8s" % (
    "key", "PSNR", "SSIM", "LPIPS", "SCORE", "dScore", "cP", "cS", "cL"))
for k in order:
    if k not in have:
        continue
    r = agg(k)
    print("%-14s %8.4f %8.5f %9.5f %9.4f | %+8.4f %+8.4f %+8.4f %+8.4f" % (
        k, r[0], r[1], r[2], r[3], r[3] - b[3],
        0.6 * (r[0] - b[0]), 30 * (r[1] - b[1]), -40 * (r[2] - b[2])))

# oracle over EVERYTHING
allk = [k for k in have]
pick = {s: max(allk, key=lambda k: sc(D[s][k]["psnr"], D[s][k]["ssim"], D[s][k]["lpips"])) for s in stems}
P = np.mean([D[s][pick[s]]["psnr"] for s in stems])
S = np.mean([D[s][pick[s]]["ssim"] for s in stems])
L = np.mean([D[s][pick[s]]["lpips"] for s in stems])
print("\nORACLE over all %d operators  PSNR %.4f SSIM %.5f LPIPS %.5f SCORE %.4f  d %+.4f" % (
    len(allk), P, S, L, sc(P, S, L), sc(P, S, L) - b[3]))
from collections import Counter
print("picks:", Counter(pick.values()).most_common())

# oracle restricted to unsharp family incl negatives
usk = ["base"] + [k for k in have if k.startswith("us_")]
pick2 = {s: max(usk, key=lambda k: sc(D[s][k]["psnr"], D[s][k]["ssim"], D[s][k]["lpips"])) for s in stems}
P2 = np.mean([D[s][pick2[s]]["psnr"] for s in stems]); S2 = np.mean([D[s][pick2[s]]["ssim"] for s in stems])
L2 = np.mean([D[s][pick2[s]]["lpips"] for s in stems])
print("ORACLE unsharp(incl blur) SCORE %.4f  d %+.4f  picks %s" % (
    sc(P2, S2, L2), sc(P2, S2, L2) - b[3], Counter(pick2.values()).most_common()))

# wiener gain curve shape
f = sorted(glob.glob(os.path.join(CACHE, "*__specw_sh.json")))
if f:
    g = np.array(json.load(open(f[0]))["gain_curve"])
    gp = np.array(json.load(open(sorted(glob.glob(os.path.join(CACHE, "*__spec_sh.json")))[0]))["gain_curve"])
    print("\nradial gain vs frequency (frame_000010, shared-channel):")
    print("  cyc/px   power-match   MSE-optimal")
    for i in range(0, 90, 5):
        print("   %.3f      %6.3f        %6.3f" % (i * 8 / 720 * 0.70711, gp[i], g[i]))

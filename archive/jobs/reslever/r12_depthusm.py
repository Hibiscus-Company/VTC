"""R12: USM helps the k4 mean but HURTS a single member.  Mechanism: pixel-mean
   ensembling attenuates HF.  Test if the optimal strength is a stable function of a
   GT-FREE measurable (the HF deficit of the mean vs its members) -> self-calibrating rule."""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
POOL = ["m31b_nolpips", "sh3", "m31b_taillpips", "gsplatB10ut8M", "gsplatB12ut8Ms7",
        "e15ceil95", "gsplatB9ut", "gsplatB4warm", "gsplatB2", "e17visnorm",
        "gsplatB1", "gsplatB8pure", "gsplatB3", "gsplatB7ppisp2", "gsplatB5affine"]
pid = [IDX[n] for n in POOL]
VIEWS = list(range(0, 60, 3))
SIG = 0.8
ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), SIG)
rng = np.random.RandomState(7)

SETS = {1: [[0]], 2: [[0, 3]], 4: [[0, 3, 6, 9], [1, 4, 7, 10]], 7: [[0, 1, 2, 3, 4, 5, 6]],
        11: [list(range(11))], 15: [list(range(15))]}

print(f"sigma={SIG}.  D = mean_member HF power / ensemble-mean HF power - 1  (GT-free)")
print(" k  set  D(GT-free)   " + "  ".join(f"a={a:.2f}" for a in ALPHAS) + "   argmax")
res = {}
for k, sets in SETS.items():
    for si, sel in enumerate(sets):
        Dacc = []
        rows = np.zeros(len(ALPHAS))
        accs = [[0., 0., 0.] for _ in ALPHAS]
        for i in VIEWS:
            ms = [R[pid[j], i].astype(np.float32) / 255.0 for j in sel]
            m = np.clip(np.mean(ms, 0), 0, 1)
            hm = (hp(m) ** 2).mean()
            hmem = np.mean([(hp(x) ** 2).mean() for x in ms])
            Dacc.append(hmem / hm - 1)
            mf = apply_field(m, field)
            g = G[i].astype(np.float32) / 255.0
            base_hp = mf - cv2.GaussianBlur(mf, (0, 0), SIG)
            for ai, a in enumerate(ALPHAS):
                p = np.clip(mf + a * base_hp, 0, 1)
                x, y, z = score(p, g)
                accs[ai][0] += x; accs[ai][1] += y; accs[ai][2] += z
        n = len(VIEWS)
        cs = [comp(a[0] / n, a[1] / n, a[2] / n) for a in accs]
        best = int(np.argmax(cs))
        res[f"k{k}_{si}"] = dict(D=float(np.mean(Dacc)), comps=cs, best_alpha=ALPHAS[best])
        print(f"{k:2d}  {si}   {np.mean(Dacc):9.4f}   " +
              "  ".join(f"{c-cs[0]:+6.3f}" for c in cs) + f"    a*={ALPHAS[best]:.2f}"
              f"  (base {cs[0]:.4f})")
json.dump(res, open(OUT + "/r12.json", "w"))

# disjoint-ensemble CV at k=4
a = res["k4_0"]; b = res["k4_1"]
ia = int(np.argmax(a["comps"])); ib = int(np.argmax(b["comps"]))
print(f"\ndisjoint k=4 CV: setA picks a={ALPHAS[ia]:.2f} -> on setB {b['comps'][ia]-b['comps'][0]:+.4f}")
print(f"                 setB picks a={ALPHAS[ib]:.2f} -> on setA {a['comps'][ib]-a['comps'][0]:+.4f}")

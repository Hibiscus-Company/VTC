"""R0: baseline scores on the production harness."""
import sys, json, numpy as np
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
print("field shape", field.shape, "mag px mean", np.abs(field).mean())

res = {}


def run(tag, get):
    P = S = L = 0.0
    per = []
    for i in range(60):
        g = G[i].astype(np.float32) / 255.0
        p, s, l = score(get(i), g)
        per.append((p, s, l))
        P += p; S += s; L += l
    P, S, L = P / 60, S / 60, L / 60
    res[tag] = dict(psnr=P, ssim=S, lpips=L, comp=comp(P, S, L), per=per)
    print(f"{tag:28s} PSNR {P:7.4f}  SSIM {S:.5f}  LPIPS {L:.5f}  comp {comp(P,S,L):8.4f}")


k4i = IDX["k4"]
run("k4_raw", lambda i: R[k4i, i].astype(np.float32) / 255.0)
run("k4_field", lambda i: apply_field(R[k4i, i].astype(np.float32) / 255.0, field))
for m in MEM:
    mi = IDX[m]
    run("mem_" + m, lambda i, mi=mi: R[mi, i].astype(np.float32) / 255.0)

json.dump(res, open(OUT + "/r0.json", "w"))

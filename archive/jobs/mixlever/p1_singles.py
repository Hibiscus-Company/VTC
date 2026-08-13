import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool as P

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    N = P.names()[:21]
    pl = P.Pool()
    print("pool ready %.1fs" % (time.time() - t0), flush=True)
    cands = {}
    for i, n in enumerate(N):
        w = np.zeros(21); w[i] = 1.0
        cands[n] = w
    UT = [N.index(x) for x in ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]]
    w = np.zeros(21); w[UT] = 0.25
    cands["_UT4_shipped_k4"] = w
    res = pl.score(cands)
    pl.close()
    json.dump(res, open(os.path.join(HERE, "p1.json"), "w"))
    rows = sorted(((P.agg(res[n])[0], n) for n in cands), reverse=True)
    print(f"{'member':22s} {'score':>8s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s}")
    for s, n in rows:
        sc, ps, ss, lp = P.agg(res[n])
        print(f"{n:22s} {sc:8.4f} {ps:7.4f} {ss:7.4f} {lp:7.4f}")
    print("elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

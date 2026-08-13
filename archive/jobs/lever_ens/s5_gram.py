"""Per-image Gram matrices X^T X, X^T y, y^T y over ALL variants (CPU, float64).
Lets us solve NNLS-MSE weights for any member subset on any image subset, exactly."""
import os, sys, numpy as np, time
from concurrent.futures import ProcessPoolExecutor

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(TMP, "lever_ens")
NAMES = open(os.path.join(TMP, "names.txt")).read().split("\n")
V = len(NAMES)


def one(gi):
    R = np.load(os.path.join(TMP, "renders_u8.npy"), mmap_mode="r")
    G = np.load(os.path.join(TMP, "gt_u8.npy"), mmap_mode="r")
    X = np.asarray(R[:, gi], dtype=np.float32).reshape(V, -1).T / 255.0   # (P, V)
    y = np.asarray(G[gi], dtype=np.float32).reshape(-1) / 255.0
    A = (X.T.astype(np.float64) @ X.astype(np.float64))
    b = X.T.astype(np.float64) @ y.astype(np.float64)
    c = float(y @ y)
    return A, b, c, X.shape[0]


if __name__ == "__main__":
    t0 = time.time()
    As = np.zeros((60, V, V)); bs = np.zeros((60, V)); cs = np.zeros(60); ns = np.zeros(60)
    with ProcessPoolExecutor(max_workers=6) as ex:
        for gi, (A, b, c, n) in enumerate(ex.map(one, range(60))):
            As[gi] = A; bs[gi] = b; cs[gi] = c; ns[gi] = n
            if gi % 10 == 0:
                print(gi, f"{time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, "gram.npz"), A=As, b=bs, c=cs, n=ns, names=np.array(NAMES))
    print("DONE", time.time() - t0)

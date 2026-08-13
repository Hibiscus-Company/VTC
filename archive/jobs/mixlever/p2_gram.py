"""Per-view Gram matrices (for NNLS weight fitting) + residual-correlation structure. CPU."""
import os, sys, time
import numpy as np

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
HERE = os.path.dirname(os.path.abspath(__file__))

R = np.load(os.path.join(TMP, "renders_u8.npy"), mmap_mode="r")
G = np.load(os.path.join(TMP, "gt_u8.npy"), mmap_mode="r")
N = open(os.path.join(TMP, "names.txt")).read().split()[:21]
NV = R.shape[1]

gram = np.zeros((NV, 21, 21)); xty = np.zeros((NV, 21)); gtg = np.zeros(NV)
rcorr = np.zeros((NV, 21, 21)); rstd = np.zeros((NV, 21)); rmean = np.zeros((NV, 21))
ddiff = np.zeros((NV, 21, 21))
t0 = time.time()
for v in range(NV):
    M = np.asarray(R[:21, v], dtype=np.float32).reshape(21, -1) / 255.0
    g = np.asarray(G[v], dtype=np.float32).reshape(-1) / 255.0
    n = M.shape[1]
    gram[v] = (M @ M.T).astype(np.float64) / n
    xty[v] = (M @ g).astype(np.float64) / n
    gtg[v] = float(np.dot(g, g)) / n
    E = M - g[None, :]
    mu = E.mean(1)
    rmean[v] = mu
    Ec = E - mu[:, None]
    C = (Ec @ Ec.T).astype(np.float64) / n
    sd = np.sqrt(np.diag(C))
    rstd[v] = sd
    rcorr[v] = C / (sd[:, None] * sd[None, :] + 1e-12)
    sq = np.diag(gram[v])
    d2 = sq[:, None] + sq[None, :] - 2 * gram[v]
    ddiff[v] = np.sqrt(np.clip(d2, 0, None))
    del M, g, E, Ec, C
    if v % 10 == 0:
        print("view", v, "%.1fs" % (time.time() - t0), flush=True)
np.savez(os.path.join(HERE, "p2_gram.npz"), gram=gram, xty=xty, gtg=gtg,
         rcorr=rcorr, rstd=rstd, rmean=rmean, ddiff=ddiff, names=np.array(N))
print("saved %.1fs" % (time.time() - t0))

"""Persistent 2-GPU worker pool: scores candidate weight-vectors, sharding views."""
import os, sys, json
import numpy as np
import multiprocessing as mp

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
NV = 60


def _worker(q_in, q_out, gpu, views):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["DEV"] = "cuda:0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import core
    core.data(); core.vgg()
    q_out.put(("ready", None))
    while True:
        job = q_in.get()
        if job is None:
            break
        tag, cands = job
        cands = {k: np.asarray(v, np.float64) for k, v in cands.items()}
        out = core.score_weights(cands, views=views, verbose=False)
        q_out.put((tag, out))


class Pool:
    def __init__(self, gpus=(0, 1), nv=NV):
        ctx = mp.get_context("spawn")
        allv = list(range(nv))
        self.shards = [allv[i::len(gpus)] for i in range(len(gpus))]
        self.qin = [ctx.Queue() for _ in gpus]
        self.qout = [ctx.Queue() for _ in gpus]
        self.ps = []
        for i, g in enumerate(gpus):
            p = ctx.Process(target=_worker, args=(self.qin[i], self.qout[i], g, self.shards[i]), daemon=True)
            p.start(); self.ps.append(p)
        for i in range(len(gpus)):
            assert self.qout[i].get()[0] == "ready"

    def score(self, cands):
        """cands: dict name -> weight vector (21,). Returns name -> per-view dict
        indexed by TRUE view index."""
        for i in range(len(self.ps)):
            self.qin[i].put(("j", cands))
        merged = {n: {"psnr": [None] * NV, "ssim": [None] * NV, "lpips": [None] * NV} for n in cands}
        for i in range(len(self.ps)):
            _, out = self.qout[i].get()
            for n, rec in out.items():
                for j, v in enumerate(self.shards[i]):
                    for k in ("psnr", "ssim", "lpips"):
                        merged[n][k][v] = rec[k][j]
        return merged

    def close(self):
        for q in self.qin:
            q.put(None)
        for p in self.ps:
            p.join(timeout=10)


def combine(P, S, L):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))


def agg(rec, views=None):
    idx = list(range(NV)) if views is None else list(views)
    P = float(np.mean([rec["psnr"][i] for i in idx]))
    S = float(np.mean([rec["ssim"][i] for i in idx]))
    L = float(np.mean([rec["lpips"][i] for i in idx]))
    return combine(P, S, L), P, S, L


def names():
    return open(os.path.join(TMP, "names.txt")).read().split()

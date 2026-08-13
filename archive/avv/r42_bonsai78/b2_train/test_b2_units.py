#!/usr/bin/env python
"""CPU-only unit tests for the B2 patch internals. NO GPU, NO training."""
import os, sys, types
import numpy as np
import torch

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/d/avv/r42_bonsai78/b2_train")
import train_gsplat_b2 as T

torch.manual_seed(0)
ok = True


def chk(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")


# NOTE: torch's CPU autograd ENGINE aborts on this box ("free(): double free detected in
# tcache 2", exit 134) in BOTH conda envs, for even `torch.autograd.grad((a**2).sum(), a)`.
# It is an environment defect, not a patch defect -- the GPU path is unaffected (training
# runs backward every step). So differentiability is verified WITHOUT running the engine:
# (a) the output carries a grad_fn tracing back to sigma (the graph is connected -- the only
# way a patch like this silently fails is a detach or an in-place break), and (b) a central
# finite difference shows the loss actually responds to sigma at a usable magnitude.
print("1. gauss_blur1: correctness, energy, differentiability w.r.t. sigma")
img = torch.rand(64, 96, 3, dtype=torch.float64)
for s in (0.3, 1.0, 2.5):
    sig = torch.tensor(s, dtype=torch.float64, requires_grad=True)
    out = T.gauss_blur1(img, sig, radius=8)
    chk(f"sigma={s} shape", out.shape == img.shape, str(tuple(out.shape)))
    chk(f"sigma={s} energy preserved", abs(float(out.mean() - img.mean())) < 2e-3,
        f"d_mean {float(out.mean()-img.mean()):+.2e}")
    chk(f"sigma={s} autograd graph reaches sigma",
        out.requires_grad and out.grad_fn is not None, f"grad_fn {type(out.grad_fn).__name__}")
    h = 1e-5
    f = lambda v: float((T.gauss_blur1(
        img, torch.tensor(v, dtype=torch.float64), 8) ** 2).sum())
    fd = (f(s + h) - f(s - h)) / (2 * h)
    chk(f"sigma={s} finite-difference dL/dsigma is nonzero", abs(fd) > 1e-6,
        f"dL/dsigma ~ {fd:+.4e}")
# blur must monotonically reduce laplacian variance
lv = []
for s in (0.05, 0.5, 1.0, 2.0):
    o = T.gauss_blur1(img, torch.tensor(s, dtype=torch.float64), 8)
    lap = (o[2:, 1:-1] + o[:-2, 1:-1] + o[1:-1, 2:] + o[1:-1, :-2] - 4 * o[1:-1, 1:-1])
    lv.append(float(lap.var()))
chk("lapvar strictly decreasing in sigma", all(a > b for a, b in zip(lv, lv[1:])),
    " -> ".join(f"{v:.3e}" for v in lv))
# agreement with an independent Gaussian reference
try:
    from scipy.ndimage import gaussian_filter
    ref = np.stack([gaussian_filter(img[..., c].numpy(), 1.0, mode="reflect") for c in range(3)], -1)
    got = T.gauss_blur1(img, torch.tensor(1.0, dtype=torch.float64), 8).numpy()
    m = np.abs(ref - got)[8:-8, 8:-8].max()
    chk("matches scipy gaussian_filter(sigma=1) in the interior", m < 2e-3, f"max|d| {m:.2e}")
except ImportError:
    print("  [SKIP] scipy not available")
# calibration: does 1 px of added sigma cost ~1 nat of log_lapvar (the A2 constant)?
big = torch.from_numpy(np.random.RandomState(0).randn(512, 512, 3) * 0.1 + 0.5)
b0 = T.gauss_blur1(big, torch.tensor(0.8, dtype=torch.float64), 8)
lvv = []
for extra in (0.0, 1.0):
    s = (0.8 ** 2 + extra ** 2) ** 0.5
    o = T.gauss_blur1(big, torch.tensor(s, dtype=torch.float64), 8)
    lap = (o[2:, 1:-1] + o[:-2, 1:-1] + o[1:-1, 2:] + o[1:-1, :-2] - 4 * o[1:-1, 1:-1])
    lvv.append(np.log(float(lap.var())))
print(f"       (white-noise probe: +1.0 px sigma costs {lvv[0]-lvv[1]:+.2f} nats of "
      f"log_lapvar; A2's photo calibration is 1.00 nats/px -- noise is the worst case, "
      f"real content is less steep)")

print("2. build_view_weights")
S = {int(r.split(",")[1]): float(r.split(",")[2])
     for r in open("/mnt/d/avv/r42_bonsai78/b2_train/bonsai_sharp_sidecar.csv").read().splitlines()[1:]}
names = [f"frame_{f:06d}.jpg" for f in sorted(S)]
sharp = {f"frame_{f:06d}.jpg": v for f, v in S.items()}
mk = lambda **kw: types.SimpleNamespace(
    **{**dict(sw_mode="tilt", sharp_stat="log_lapvar", sw_power=2.0, sw_beta=0.5,
              sw_topk=0.5, sw_topk_stratify=0, sw_wmin=0.05, sw_wmax=5.0), **kw})
for kw in (dict(sw_mode="tilt", sw_beta=0.5), dict(sw_mode="tilt", sw_beta=1.0),
           dict(sw_mode="power", sw_power=2.0),
           dict(sw_mode="topk", sw_topk=0.5),
           dict(sw_mode="topk", sw_topk=0.5, sw_topk_stratify=6)):
    w = T.build_view_weights(mk(**kw), names, sharp)
    chk(f"  {kw} mean==1", abs(w.mean() - 1) < 1e-9, f"mean {w.mean():.6f}")
    chk(f"  {kw} nonneg", (w >= 0).all())
# stratified topk must preserve temporal coverage
w = T.build_view_weights(mk(sw_mode="topk", sw_topk=0.5, sw_topk_stratify=6), names, sharp)
kept = np.array(sorted(S))[w > 0]
wu = T.build_view_weights(mk(sw_mode="topk", sw_topk=0.5, sw_topk_stratify=0), names, sharp)
keptu = np.array(sorted(S))[wu > 0]
chk("stratified topk max gap << unstratified", np.diff(kept).max() < np.diff(keptu).max() / 3,
    f"strat {np.diff(kept).max()} vs unstrat {np.diff(keptu).max()}")

print("3. blur_sigma_init")
for q in (0.5, 0.75, 0.95):
    a = types.SimpleNamespace(sharp_stat="log_lapvar", blur_ref_q=q, blur_gain=1.0,
                              blur_sigma_max=2.5)
    sg = T.blur_sigma_init(a, names, sharp)
    chk(f"  q={q} zero-fraction ~= 1-q", abs((sg <= 1e-6).mean() - (1 - q)) < 0.02,
        f"{(sg<=1e-6).mean():.3f} vs {1-q:.3f}")
    chk(f"  q={q} within cap", sg.max() <= 2.5 + 1e-9)

print("4. sigmoid parameterisation round-trip (learn mode init)")
FL, MX = 0.05, 2.5
a = types.SimpleNamespace(sharp_stat="log_lapvar", blur_ref_q=0.75, blur_gain=1.0,
                          blur_sigma_max=MX)
sg0 = torch.tensor(T.blur_sigma_init(a, names, sharp), dtype=torch.float32)
u = ((sg0 - FL) / (MX - FL)).clamp(1e-4, 1 - 1e-4)
raw = torch.logit(u)
back = FL + (MX - FL) * torch.sigmoid(raw)
err = float((back - sg0.clamp(min=FL)).abs().max())
chk("logit/sigmoid round-trip", err < 1e-3, f"max err {err:.2e} px "
    f"(views initialised below the {FL} px floor are clamped up to it, by design)")

print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)

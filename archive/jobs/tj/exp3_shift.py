"""EXP-3  TRAJECTORY TRANSFER of a low-dimensional GEOMETRIC correction.

The named lever: "can a test render be improved using the two temporally-bracketing TRAIN
renders in a way that is NOT the failed IBR photo-warp?"  We transfer TWO NUMBERS (a global
sub-pixel translation), not pixels -- so it is neither IBR nor the per-view dense field that
was shown to be a mirage (that one had H*W*2 free params per view and fit its own DIS noise;
this has 2, estimated from 240-photo-strong train data).

MEASURED CONTEXT: all 60 test views are temporally bracketed by train views and 57/60 have a
train frame at distance 1 in the capture sequence.

Ladder
  base                     k4 float pixel-mean, no field
  GLOBAL                   one (dx,dy) for all views, fit on TRAIN renders vs TRAIN photos
                           (this is the translation component of the shipped median lens field)
  ORACLE-view              per-view (dx,dy) fit against the TEST GT  -> upper bound
  HONEST-view              per-view (dx,dy) predicted from bracketing TRAIN views only
Plus a leave-one-out CV on the train views: can a train view's own shift be predicted from
its trajectory neighbours at all?
"""
import os, re, json, bisect
import numpy as np, torch, torch.nn.functional as F
import hlib
from PIL import Image

DEV = 'cuda'
PUB = '/mnt/d/avv/data/phase1/public_set/HCM0181'
OUT = '/mnt/d/avv/output'
gi = lambda f: int(re.search(r'_(\d{4})_V', f).group(1))


def shift_img(t, dx, dy):
    """t [1,3,H,W]; positive dx samples from the right -> content moves LEFT."""
    _, _, H, W = t.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=t.device, dtype=torch.float32),
                            torch.arange(W, device=t.device, dtype=torch.float32),
                            indexing='ij')
    gx = ((xx + dx) / (W - 1)) * 2 - 1
    gy = ((yy + dy) / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(t, grid, mode='bicubic', padding_mode='border', align_corners=True)


def fit_shift(r, g, iters=4):
    """MSE-optimal global translation taking r -> g (Lucas-Kanade, sub-pixel)."""
    dx = dy = 0.0
    for _ in range(iters):
        w = shift_img(r, dx, dy)
        gxk = torch.tensor([[[[-.5, 0, .5]]]], device=r.device).repeat(3, 1, 1, 1)
        gyk = gxk.transpose(-1, -2)
        Ix = F.conv2d(F.pad(w, (1, 1, 0, 0), mode='replicate'), gxk, groups=3)
        Iy = F.conv2d(F.pad(w, (0, 0, 1, 1), mode='replicate'), gyk, groups=3)
        e = g - w
        m = torch.zeros_like(e[:, :1]); m[..., 8:-8, 8:-8] = 1     # ignore border
        Ix, Iy, e = Ix * m, Iy * m, e * m
        A = torch.tensor([[(Ix * Ix).sum(), (Ix * Iy).sum()],
                          [(Ix * Iy).sum(), (Iy * Iy).sum()]], dtype=torch.float64)
        b = torch.tensor([(Ix * e).sum(), (Iy * e).sum()], dtype=torch.float64)
        try:
            s = torch.linalg.solve(A, b)
        except Exception:
            break
        # w(x)=r(x+dx); w_new = w + ddx*Ix + ddy*Iy  ->  step ADDS to the grid offset
        dx += float(s[0]); dy += float(s[1])
        if abs(float(s[0])) < 1e-4 and abs(float(s[1])) < 1e-4:
            break
    return dx, dy


def _selftest():
    """Recover a known synthetic translation; guards the sign convention."""
    g = torch.rand(1, 3, 96, 128, device=DEV)
    g = F.avg_pool2d(F.pad(g, (2, 2, 2, 2), mode='replicate'), 5, 1)   # smooth -> LK-valid
    for (tx, ty) in [(0.30, -0.40), (-0.75, 0.25)]:
        r = shift_img(g, -tx, -ty)          # r is g displaced; fitting r->g must return (tx,ty)
        dx, dy = fit_shift(r, g)
        assert abs(dx - tx) < 0.05 and abs(dy - ty) < 0.05, \
            f"LK selftest FAILED: wanted ({tx},{ty}) got ({dx:.3f},{dy:.3f})"
    print("  LK selftest OK (sign convention verified)")


def t_of(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)
                            ).permute(2, 0, 1).unsqueeze(0).to(DEV)


hlib.init()
N = hlib.names(); R = hlib.renders(); GU8 = np.asarray(hlib.gt())
MEM = ['gsplatB9ut', 'gsplatB10ut8M', 'gsplatB11ut60k', 'gsplatB12ut8Ms7']
base = np.zeros(GU8.shape, np.float32)
for m in MEM:
    base += R[N.index(m)].astype(np.float32) / 255.0
base /= len(MEM)
n = len(base)
te_files = sorted(os.listdir(f'{PUB}/test/images'))
TE = [gi(f) for f in te_files]

# ================= per-TEST-view ORACLE shifts =================
print("=== fitting per-view ORACLE shifts (against TEST GT) ===", flush=True)
_selftest()
osh = []
worse = 0
for i in range(n):
    r = t_of(base[i]); g = t_of(GU8[i].astype(np.float32) / 255.0)
    d = fit_shift(r, g)
    # an ORACLE fit must not increase MSE -- if it does the fit diverged, fall back to 0
    m0 = float(((r - g) ** 2).mean())
    m1 = float(((shift_img(r, d[0], d[1]) - g) ** 2).mean())
    if m1 > m0:
        worse += 1; d = (0.0, 0.0)
    osh.append(d)
osh = np.array(osh)
print(f"  fits that had to fall back to zero: {worse}/{n}")
print(f"  oracle shift  dx mean {osh[:,0].mean():+.4f} std {osh[:,0].std():.4f} | "
      f"dy mean {osh[:,1].mean():+.4f} std {osh[:,1].std():.4f}  (px)")

# ================= TRAIN-view shifts (honest source) =================
print("\n=== fitting shifts on TRAIN views (render vs train photo) ===", flush=True)
trd = f'{OUT}/HCM0181_gsplatB9ut/train_renders'
tr_files = sorted(os.listdir(trd))
TRI = [gi(f) for f in tr_files]
tsh = []
for f in tr_files:
    rp = np.asarray(Image.open(os.path.join(trd, f)).convert('RGB'), np.float32) / 255.
    gp = np.asarray(Image.open(f'{PUB}/train/images/{os.path.splitext(f)[0]}.JPG'
                               ).convert('RGB'), np.float32) / 255.
    tsh.append(fit_shift(t_of(rp), t_of(gp)))
tsh = np.array(tsh)
print(f"  train  shift  dx mean {tsh[:,0].mean():+.4f} std {tsh[:,0].std():.4f} | "
      f"dy mean {tsh[:,1].mean():+.4f} std {tsh[:,1].std():.4f}  (px)  n={len(tsh)}")

# ================= is the per-view shift TEMPORALLY STRUCTURED? =================
print("\n=== is the per-view residual shift predictable from neighbours? ===")
o = np.argsort(TRI); Ts = np.array(TRI)[o]; Ss = tsh[o]
loo_err, base_err = [], []
for k in range(len(Ts)):
    nb = [j for j in range(len(Ts)) if j != k]
    nb.sort(key=lambda j: abs(Ts[j] - Ts[k]))
    pred = Ss[nb[:2]].mean(0)
    loo_err.append(((pred - Ss[k]) ** 2).sum())
    base_err.append((((np.delete(Ss, k, 0)).mean(0) - Ss[k]) ** 2).sum())
loo_err, base_err = np.mean(loo_err), np.mean(base_err)
print(f"  LOO   neighbour-predicted shift  MSE {loo_err:.5f} px^2")
print(f"  LOO   global-mean    shift  MSE {base_err:.5f} px^2")
print(f"  ==> neighbours beat the global mean by {(1-loo_err/base_err)*100:+.1f}%  "
      f"({'STRUCTURED' if loo_err < base_err else 'NO STRUCTURE'})")

# ================= build the correction variants =================
gm = tsh.mean(0)                                    # honest global shift from TRAIN
print(f"\n  honest GLOBAL shift from train = ({gm[0]:+.4f},{gm[1]:+.4f}) px")
hon = []
for t in TE:
    nb = sorted(range(len(TRI)), key=lambda j: abs(TRI[j] - t))[:2]
    hon.append(tsh[nb].mean(0))
hon = np.array(hon)
print(f"  honest per-view predicted shift std ({hon[:,0].std():.4f},{hon[:,1].std():.4f}) px")
print(f"  corr(honest, oracle)  dx {np.corrcoef(hon[:,0],osh[:,0])[0,1]:+.3f}  "
      f"dy {np.corrcoef(hon[:,1],osh[:,1])[0,1]:+.3f}")


def apply(sh):
    out = np.empty_like(base)
    for i in range(n):
        w = shift_img(t_of(base[i]), float(sh[i][0]), float(sh[i][1]))
        out[i] = w[0].permute(1, 2, 0).cpu().numpy()
    return out


print("\n=== production-harness scores ===")
res = {}
res['base'] = hlib.score(base, GU8, "BASE k4 float-mean")
res['glob'] = hlib.score(apply(np.repeat(gm[None], n, 0)), GU8, "GLOBAL shift (train-fit)")
res['orac'] = hlib.score(apply(osh), GU8, "ORACLE per-view shift")
res['hon'] = hlib.score(apply(hon), GU8, "HONEST per-view (train nbrs)")
# oracle per-view on top of the honest global (isolates the *extra* per-view part)
res['orac_x'] = hlib.score(apply(osh - gm + gm), GU8, "ORACLE per-view (abs)  [dup chk]")

b = res['base']['score']
print("\n%-34s %8s %9s" % ("config", "score", "dScore"))
for k, v in res.items():
    print("%-34s %8.4f  %+8.4f" % (v['tag'], v['score'], v['score'] - b))
np.save('/home/bkai/.claude/jobs/1c9cf7e9/tmp/tj/shifts.npy',
        {'oracle': osh, 'train': tsh, 'tri': TRI, 'te': TE, 'honest': hon}, allow_pickle=True)
json.dump({k: v for k, v in res.items()},
          open('/home/bkai/.claude/jobs/1c9cf7e9/tmp/tj/exp3.json', 'w'), indent=1)

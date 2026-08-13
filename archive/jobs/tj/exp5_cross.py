"""EXP-5  CROSS-SCENE validation of the one surviving lever (EXP-3 per-view shift transfer).

EXP-3 found +0.0177 honest on HCM0181.  A single-scene +0.018 is exactly the size of result
that has burned this project before, so it gets tested on the four other production scenes
before any verdict.  Single member (gsplatB9ut) here -- these scenes have no ensemble -- so
this measures the lever's own sign, not its size in the k6 pipeline.

Honest estimator only: shifts fit on TRAIN renders vs TRAIN photos, transferred to the test
view from its two trajectory-nearest TRAIN frames.  No test GT touches the estimator.
"""
import os, re, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import hlib

DEV = 'cuda'
PUB = '/mnt/d/avv/data/phase1/public_set'
OUT = '/mnt/d/avv/output'
Image.MAX_IMAGE_PIXELS = None
gi = lambda f: int(re.search(r'_(\d+)_', f).group(1)) if re.search(r'_(\d+)_', f) \
    else int(re.search(r'(\d+)', f).group(1))


def shift_img(t, dx, dy):
    _, _, H, W = t.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=t.device, dtype=torch.float32),
                            torch.arange(W, device=t.device, dtype=torch.float32),
                            indexing='ij')
    grid = torch.stack([((xx + dx) / (W - 1)) * 2 - 1,
                        ((yy + dy) / (H - 1)) * 2 - 1], -1).unsqueeze(0)
    return F.grid_sample(t, grid, mode='bicubic', padding_mode='border', align_corners=True)


def fit_shift(r, g, iters=4):
    dx = dy = 0.0
    gxk = torch.tensor([[[[-.5, 0, .5]]]], device=r.device).repeat(3, 1, 1, 1)
    gyk = gxk.transpose(-1, -2)
    for _ in range(iters):
        w = shift_img(r, dx, dy)
        Ix = F.conv2d(F.pad(w, (1, 1, 0, 0), mode='replicate'), gxk, groups=3)
        Iy = F.conv2d(F.pad(w, (0, 0, 1, 1), mode='replicate'), gyk, groups=3)
        e = g - w
        m = torch.zeros_like(e[:, :1]); m[..., 8:-8, 8:-8] = 1
        Ix, Iy, e = Ix * m, Iy * m, e * m
        A = torch.tensor([[(Ix * Ix).sum(), (Ix * Iy).sum()],
                          [(Ix * Iy).sum(), (Iy * Iy).sum()]], dtype=torch.float64)
        b = torch.tensor([(Ix * e).sum(), (Iy * e).sum()], dtype=torch.float64)
        s = torch.linalg.solve(A, b)
        dx += float(s[0]); dy += float(s[1])
        if abs(float(s[0])) < 1e-4 and abs(float(s[1])) < 1e-4:
            break
    return dx, dy


def ld(p):
    return np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.


def t_of(a):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32)).permute(2, 0, 1).unsqueeze(0).to(DEV)


hlib.init()
rows = []
for sc in ['HCM0193', 'HCM0204', 'hcm0031', 'hcm0034']:
    rd = f'{OUT}/{sc}_gsplatB9ut/test_poses_renders_png'
    trd = f'{OUT}/{sc}_gsplatB9ut/train_renders'
    tef = sorted(os.listdir(rd))
    gtmap = {os.path.splitext(f)[0]: f for f in os.listdir(f'{PUB}/{sc}/test/images')}
    trf = sorted(os.listdir(trd))
    trg = {os.path.splitext(f)[0]: f for f in os.listdir(f'{PUB}/{sc}/train/images')}

    tsh, TRI = [], []
    for f in trf:
        s = os.path.splitext(f)[0]
        if s not in trg:
            continue
        tsh.append(fit_shift(t_of(ld(os.path.join(trd, f))),
                             t_of(ld(f'{PUB}/{sc}/train/images/{trg[s]}'))))
        TRI.append(gi(s))
    tsh = np.array(tsh); TRI = np.array(TRI)

    base, corr, GT = [], [], []
    for f in tef:
        s = os.path.splitext(f)[0]
        r = ld(os.path.join(rd, f))
        base.append(r); GT.append(np.asarray(Image.open(f'{PUB}/{sc}/test/images/{gtmap[s]}'
                                                        ).convert('RGB'), np.uint8))
        t = gi(s)
        nb = np.argsort(np.abs(TRI - t))[:2]
        d = tsh[nb].mean(0)
        corr.append(shift_img(t_of(r), float(d[0]), float(d[1]))[0].permute(1, 2, 0).cpu().numpy())
    base = np.stack(base); corr = np.stack(corr); GT = np.stack(GT)
    a = hlib.score(base, GT, f"[{sc}] base")
    b = hlib.score(corr, GT, f"[{sc}] +shift transfer")
    print(f"   train shift std ({tsh[:,0].std():.4f},{tsh[:,1].std():.4f}) px   "
          f"dScore {b['score']-a['score']:+.4f}\n", flush=True)
    rows.append((sc, a['score'], b['score'], b['score'] - a['score']))

print("%-12s %9s %9s %9s" % ("scene", "base", "+shift", "dScore"))
for sc, a, b, d in rows:
    print("%-12s %9.4f %9.4f %+9.4f" % (sc, a, b, d))
print("%-12s %9s %9s %+9.4f" % ("MEAN", "", "", np.mean([r[3] for r in rows])))
print("%-12s %9s %9s %9s" % ("sign", "", "", f"{sum(1 for r in rows if r[3]>0)}/{len(rows)} positive"))
json.dump(rows, open('/home/bkai/.claude/jobs/1c9cf7e9/tmp/tj/exp5.json', 'w'), indent=1)

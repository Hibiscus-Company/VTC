"""(a) Local-contrast gain: global vs BORDER-ONLY vs INTERIOR-ONLY, both directions.
Full view count. Variants span a range of render smoothness (single -> ensemble -> ensemble+field)
so we can read the break-even vs correlation rho and place production on that axis.
Split-half CV on the fitted gain."""
import sys, os, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from utils.loss_utils import create_window

VAR = [('HCM0181_B9ut', f'{OUT}/HCM0181_gsplatB9ut/test_poses_renders_png', 'HCM0181'),
       ('HCM0181_k4',   '/mnt/d/avv/prodharness/k4/png', 'HCM0181'),
       ('HCM0181_k4f',  '/mnt/d/avv/prodharness/k4f', 'HCM0181'),
       ('HCM0193_B9ut', f'{OUT}/HCM0193_gsplatB9ut/test_poses_renders_png', 'HCM0193')]
SIG = 1.0
GAINS = [0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20]
BW = 20  # border band width for border-only ops

def gk(sig, dev):
    r = max(1, int(round(3*sig))); k = torch.arange(-r, r+1, device=dev, dtype=torch.float32)
    g = torch.exp(-k*k/(2*sig*sig)); return g/g.sum(), r

def blur(x, sig):
    g, r = gk(sig, x.device)
    w1 = g[None, None, None, :].expand(3, 1, 1, 2*r+1)
    w2 = g[None, None, :, None].expand(3, 1, 2*r+1, 1)
    z = F.conv2d(F.pad(x[None], (r, r, 0, 0), mode='reflect'), w1, groups=3)
    return F.conv2d(F.pad(z, (0, 0, r, r), mode='reflect'), w2, groups=3)[0]

def bandmask(H, W, dev, lo, hi):
    ii = torch.arange(H, device=dev)[:, None].expand(H, W)
    jj = torch.arange(W, device=dev)[None, :].expand(H, W)
    d = torch.minimum(torch.minimum(ii, H-1-ii), torch.minimum(jj, W-1-jj))
    return ((d >= lo) & (d < hi)).float()[None]

CFG = [('base', 'g', 1.0)]
CFG += [(f'glob_g{g}', 'g', g) for g in GAINS]
CFG += [(f'bord_g{g}', 'b', g) for g in GAINS]
CFG += [(f'intr_g{g}', 'i', g) for g in GAINS]

allres = {}
for name, rd, scene in VAR:
    st = stems(rd)
    per = {c[0]: [] for c in CFG}   # per-view (psnr, ssim, lpips)
    rho_acc = [0., 0., 0.]
    for si in st:
        x0 = load(f'{rd}/{si}.png'); y = load(f'{PUB}/{scene}/test/images/{si}.JPG')
        C, H, W = x0.shape
        mu = blur(x0, SIG); hp = x0 - mu
        mb = bandmask(H, W, x0.device, 0, BW)
        # rho bookkeeping (11x11 gaussian, whole frame)
        w = create_window(11, 3).type_as(x0); p = 5
        m1 = F.conv2d(x0[None], w, padding=p, groups=3); m2 = F.conv2d(y[None], w, padding=p, groups=3)
        s1 = (F.conv2d(x0[None]*x0[None], w, padding=p, groups=3)-m1*m1).mean().item()
        s2 = (F.conv2d(y[None]*y[None], w, padding=p, groups=3)-m2*m2).mean().item()
        s12 = (F.conv2d(x0[None]*y[None], w, padding=p, groups=3)-m1*m2).mean().item()
        rho_acc[0] += s1; rho_acc[1] += s2; rho_acc[2] += s12
        for cn, kind, g in CFG:
            if g == 1.0 and kind == 'g':
                x = x0
            else:
                if kind == 'g':   w_ = 1.0
                elif kind == 'b': w_ = mb
                else:             w_ = 1.0 - mb
                x = (mu + (1 + (g-1)*w_) * hp).clamp(0, 1)
            per[cn].append((psnr(x, y), ssim_val(x, y), lpips_val(x, y)))
    n = len(st)
    def sc(cn, idx=None):
        v = per[cn] if idx is None else [per[cn][i] for i in idx]
        m = len(v)
        return score(sum(a[0] for a in v)/m, sum(a[1] for a in v)/m, sum(a[2] for a in v)/m)
    A = list(range(0, n, 2)); B = list(range(1, n, 2))
    b_all, b_A, b_B = sc('base'), sc('base', A), sc('base', B)
    rec = dict(n=n, rho=rho_acc[2]/((rho_acc[0]*rho_acc[1])**0.5), base=b_all, cfg={})
    for cn, _, _ in CFG:
        rec['cfg'][cn] = dict(d=sc(cn)-b_all, dA=sc(cn, A)-b_A, dB=sc(cn, B)-b_B,
                              psnr=sum(a[0] for a in per[cn])/n,
                              ssim=sum(a[1] for a in per[cn])/n,
                              lpips=sum(a[2] for a in per[cn])/n)
    allres[name] = rec
    print(f"\n{name}  n={n}  rho={rec['rho']:.4f}  base={b_all:.4f}", flush=True)
    for cn, _, _ in CFG:
        r = rec['cfg'][cn]
        print(f"  {cn:12s} d={r['d']:+.4f}  (halfA {r['dA']:+.4f} / halfB {r['dB']:+.4f})  "
              f"P={r['psnr']:6.3f} S={r['ssim']:.4f} L={r['lpips']:.5f}", flush=True)
json.dump(allres, open(os.path.dirname(__file__)+'/a5_gain.json', 'w'), indent=1)

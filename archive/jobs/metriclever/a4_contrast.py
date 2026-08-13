"""(a)+(b) DECISIVE TEST: local-contrast gain  x' = mu + a*(x-mu)  (unsharp), global and
border-ramped, on the production harness. Coarse grid, subset of views per scene."""
import sys, os, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *

SETS = [('HCM0181_k4f', '/mnt/d/avv/prodharness/k4f', 'HCM0181'),
        ('HCM0193', f'{OUT}/HCM0193_gsplatB9ut/test_poses_renders_png', 'HCM0193'),
        ('HCM0204', f'{OUT}/HCM0204_gsplatB9ut/test_poses_renders_png', 'HCM0204'),
        ('hcm0031', f'{OUT}/hcm0031_gsplatB9ut/test_poses_renders_png', 'hcm0031'),
        ('hcm0034', f'{OUT}/hcm0034_gsplatB9ut/test_poses_renders_png', 'hcm0034')]
NSUB = int(os.environ.get('NSUB', '20'))

def gk(sig, dev):
    r = max(1, int(round(3*sig))); k = torch.arange(-r, r+1, device=dev, dtype=torch.float32)
    g = torch.exp(-k*k/(2*sig*sig)); return g/g.sum(), r

def blur(x, sig):
    g, r = gk(sig, x.device)
    w1 = g[None, None, None, :].expand(3, 1, 1, 2*r+1)
    w2 = g[None, None, :, None].expand(3, 1, 2*r+1, 1)
    z = F.conv2d(F.pad(x[None], (r, r, 0, 0), mode='reflect'), w1, groups=3)
    return F.conv2d(F.pad(z, (0, 0, r, r), mode='reflect'), w2, groups=3)[0]

SIGS = [1.0, 1.5, 2.5]
AMTS = [0.05, 0.10, 0.15, 0.20, 0.30]
CFG = [('base', None, 0.0)] + [(f's{s}_a{a}', s, a) for s in SIGS for a in AMTS]

res = {}
for name, rd, scene in SETS:
    st = stems(rd)
    st = st[::max(1, len(st)//NSUB)][:NSUB]
    acc = {c[0]: [0., 0., 0.] for c in CFG}
    for si in st:
        x0 = load(f'{rd}/{si}.png'); y = load(f'{PUB}/{scene}/test/images/{si}.JPG')
        cache = {}
        for cn, sig, amt in CFG:
            if sig is None:
                x = x0
            else:
                if sig not in cache: cache[sig] = blur(x0, sig)
                x = (x0 + amt*(x0-cache[sig])).clamp(0, 1)
            a = acc[cn]
            a[0] += psnr(x, y); a[1] += ssim_val(x, y); a[2] += lpips_val(x, y)
    n = len(st)
    b = acc['base']; bs = score(b[0]/n, b[1]/n, b[2]/n)
    res[name] = {}
    for cn, _, _ in CFG:
        a = acc[cn]; s = score(a[0]/n, a[1]/n, a[2]/n)
        res[name][cn] = dict(psnr=a[0]/n, ssim=a[1]/n, lpips=a[2]/n, score=s, d=s-bs)
    print(name, 'n=', n, flush=True)
    for cn, _, _ in CFG:
        r = res[name][cn]
        print(f"  {cn:12s} P={r['psnr']:6.3f} S={r['ssim']:.4f} L={r['lpips']:.5f} "
              f"score={r['score']:8.4f}  d={r['d']:+.4f}", flush=True)

print('\n=== dScore summary (rows=config, cols=scene) ===')
hdr = [s[0] for s in SETS]
print(f"{'config':12s} " + ' '.join(f'{h:>13s}' for h in hdr) + f" {'MEAN':>9s}")
for cn, _, _ in CFG:
    ds = [res[h][cn]['d'] for h in hdr]
    print(f"{cn:12s} " + ' '.join(f'{d:+13.4f}' for d in ds) + f" {sum(ds)/len(ds):+9.4f}")
json.dump(res, open(os.path.dirname(__file__)+'/a4_contrast.json', 'w'), indent=1)

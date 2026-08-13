"""(a) ORACLE ceiling: paste GT into the outer-k ring of the render. Bounds EVERY possible
legal border treatment (no real method can beat pasting the truth). Full composite delta.
Also: radial profile of SSIM & squared error vs distance-from-border. (c) per-image PSNR max."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *

SETS = [('HCM0181_k4f', '/mnt/d/avv/prodharness/k4f', 'HCM0181'),
        ('HCM0181_k4',  '/mnt/d/avv/prodharness/k4/png', 'HCM0181'),
        ('HCM0193_B9ut', f'{OUT}/HCM0193_gsplatB9ut/test_poses_renders_png', 'HCM0193'),
        ('hcm0034_B9ut', f'{OUT}/hcm0034_gsplatB9ut/test_poses_renders_png', 'hcm0034')]
KS = [5, 10, 20, 40]
BANDS = [(0,5),(5,10),(10,20),(20,40),(40,80),(80,160),(160,10**6)]

out = {}
for name, rd, scene in SETS:
    st = stems(rd)
    base = dict(p=0., s=0., l=0.)
    ork = {k: dict(p=0., s=0., l=0.) for k in KS}
    band_ss = {b: 0. for b in BANDS}; band_se = {b: 0. for b in BANDS}; band_n = {b: 0 for b in BANDS}
    pmax = -1e9
    for si in st:
        gp = f'{PUB}/{scene}/test/images/{si}.JPG'
        x = load(f'{rd}/{si}.png'); y = load(gp)
        p0 = psnr(x, y); pmax = max(pmax, p0)
        base['p'] += p0; base['s'] += ssim_val(x, y); base['l'] += lpips_val(x, y)
        C, H, W = x.shape
        # distance to nearest border
        ii = torch.arange(H, device=x.device)[:, None].expand(H, W)
        jj = torch.arange(W, device=x.device)[None, :].expand(H, W)
        d = torch.minimum(torch.minimum(ii, H - 1 - ii), torch.minimum(jj, W - 1 - jj))
        m = ssim_map(x, y).mean(0)
        se = ((x - y) ** 2).mean(0)
        for b in BANDS:
            msk = (d >= b[0]) & (d < b[1])
            if msk.any():
                band_ss[b] += m[msk].mean().item(); band_se[b] += se[msk].mean().item()
                band_n[b] += 1
        for k in KS:
            z = x.clone()
            z[:, :k, :] = y[:, :k, :]; z[:, -k:, :] = y[:, -k:, :]
            z[:, :, :k] = y[:, :, :k]; z[:, :, -k:] = y[:, :, -k:]
            ork[k]['p'] += psnr(z, y); ork[k]['s'] += ssim_val(z, y); ork[k]['l'] += lpips_val(z, y)
    n = len(st)
    b0 = score(base['p']/n, base['s']/n, base['l']/n)
    rec = dict(n=n, psnr=base['p']/n, ssim=base['s']/n, lpips=base['l']/n, score=b0,
               psnr_max_single=pmax,
               bands={f'{b[0]}-{b[1]}': dict(ssim=band_ss[b]/max(band_n[b],1),
                                             rmse=(band_se[b]/max(band_n[b],1))**0.5)
                      for b in BANDS},
               oracle={})
    for k in KS:
        sk = score(ork[k]['p']/n, ork[k]['s']/n, ork[k]['l']/n)
        rec['oracle'][k] = dict(psnr=ork[k]['p']/n, ssim=ork[k]['s']/n, lpips=ork[k]['l']/n,
                                score=sk, dScore=sk-b0)
    out[name] = rec
    print(name, json.dumps(rec, indent=1), flush=True)
json.dump(out, open(os.path.dirname(__file__)+'/a2_border_oracle.json','w'), indent=1)

"""(a) WHICH local statistic is mismatched near the frame border?
Per distance-band: brightness bias, local-std ratio s1/s2, correlation rho.
These identify which legal knob (offset / contrast / nothing) could recover the border deficit."""
import sys, os, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from utils.loss_utils import create_window

SETS = [('HCM0181_k4f', '/mnt/d/avv/prodharness/k4f', 'HCM0181'),
        ('HCM0193_B9ut', f'{OUT}/HCM0193_gsplatB9ut/test_poses_renders_png', 'HCM0193'),
        ('HCM0204_B9ut', f'{OUT}/HCM0204_gsplatB9ut/test_poses_renders_png', 'HCM0204'),
        ('hcm0031_B9ut', f'{OUT}/hcm0031_gsplatB9ut/test_poses_renders_png', 'hcm0031'),
        ('hcm0034_B9ut', f'{OUT}/hcm0034_gsplatB9ut/test_poses_renders_png', 'hcm0034'),
        ('HCM0181_TRAINVIEWS', f'{OUT}/HCM0181_gsplatB9ut/train_renders', 'HCM0181_TRAIN')]
BANDS = [(5,10),(10,20),(20,40),(40,80),(80,160),(160,10**6)]
W = 11; P = 5
out = {}
for name, rd, scene in SETS:
    st = stems(rd)
    if scene.endswith('_TRAIN'):
        gdir = f'{PUB}/{scene[:-6]}/train/images'
    else:
        gdir = f'{PUB}/{scene}/test/images'
    st = [s for s in st if os.path.exists(f'{gdir}/{s}.JPG')]
    agg = {b: dict(dm=0., s1=0., s2=0., s12=0., n=0) for b in BANDS}
    for si in st:
        x = load(f'{rd}/{si}.png'); y = load(f'{gdir}/{si}.JPG')
        if x.shape != y.shape: continue
        C, H, Wd = x.shape
        w = create_window(W, C).type_as(x).to(x.device)
        mu1 = F.conv2d(x[None], w, padding=P, groups=C)
        mu2 = F.conv2d(y[None], w, padding=P, groups=C)
        s1 = (F.conv2d(x[None]*x[None], w, padding=P, groups=C) - mu1*mu1)[0].mean(0)
        s2 = (F.conv2d(y[None]*y[None], w, padding=P, groups=C) - mu2*mu2)[0].mean(0)
        s12 = (F.conv2d(x[None]*y[None], w, padding=P, groups=C) - mu1*mu2)[0].mean(0)
        dm = (mu1 - mu2)[0].mean(0)
        ii = torch.arange(H, device=x.device)[:, None].expand(H, Wd)
        jj = torch.arange(Wd, device=x.device)[None, :].expand(H, Wd)
        d = torch.minimum(torch.minimum(ii, H-1-ii), torch.minimum(jj, Wd-1-jj))
        for b in BANDS:
            m = (d >= b[0]) & (d < b[1])
            if not m.any(): continue
            a = agg[b]
            a['dm'] += dm[m].mean().item(); a['s1'] += s1[m].mean().item()
            a['s2'] += s2[m].mean().item(); a['s12'] += s12[m].mean().item(); a['n'] += 1
    rec = {}
    for b in BANDS:
        a = agg[b]; n = max(a['n'], 1)
        S1, S2, S12 = a['s1']/n, a['s2']/n, a['s12']/n
        rec[f'{b[0]}-{b[1]}'] = dict(dmean=a['dm']/n, std_ratio=(S1/S2)**0.5,
                                     rho=S12/max((S1*S2)**0.5, 1e-12),
                                     a_ssim_opt=(S2/S1)**0.5, a_mse_opt=S12/S1)
    out[name] = dict(n=len(st), bands=rec)
    print(name, 'n=', len(st))
    for k, v in rec.items():
        print(f"  {k:12s} dmean={v['dmean']:+.5f}  s1/s2={v['std_ratio']:.4f}  rho={v['rho']:.4f}"
              f"  a*_ssim={v['a_ssim_opt']:.4f}  a*_mse={v['a_mse_opt']:.4f}")
json.dump(out, open(os.path.dirname(__file__)+'/a3_profile.json','w'), indent=1)

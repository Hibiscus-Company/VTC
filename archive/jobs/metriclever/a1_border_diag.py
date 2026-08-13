"""(a) How much of the SSIM score is decided by the outer 5-px ring?  (c) PSNR clamp check."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *

RENDERS = {
    'HCM0181_k4':      ('/mnt/d/avv/prodharness/k4/png', 'HCM0181'),
    'HCM0181_k4f':     ('/mnt/d/avv/prodharness/k4f',    'HCM0181'),
    'HCM0181_B9ut':    (f'{OUT}/HCM0181_gsplatB9ut/test_poses_renders_png', 'HCM0181'),
    'HCM0193_B9ut':    (f'{OUT}/HCM0193_gsplatB9ut/test_poses_renders_png', 'HCM0193'),
    'HCM0204_B9ut':    (f'{OUT}/HCM0204_gsplatB9ut/test_poses_renders_png', 'HCM0204'),
    'hcm0031_B9ut':    (f'{OUT}/hcm0031_gsplatB9ut/test_poses_renders_png', 'hcm0031'),
    'hcm0034_B9ut':    (f'{OUT}/hcm0034_gsplatB9ut/test_poses_renders_png', 'hcm0034'),
}
R = 5  # half-window: ring where the 11x11 window touches zero padding
res = {}
for name, (rd, scene) in RENDERS.items():
    if not os.path.isdir(rd):
        print('MISSING', rd); continue
    st = stems(rd)
    acc = dict(full=0., ring=0., inner=0., psnr=0., nring=0, ninner=0,
               ring_gtpad=0., inner_gtpad=0., n=0)
    for s in st:
        gp = f'{PUB}/{scene}/test/images/{s}.JPG'
        if not os.path.exists(gp):
            gp = f'{PUB}/{scene}/test/images/{s}.jpg'
        x = load(f'{rd}/{s}.png'); y = load(gp)
        m = ssim_map(x, y)
        C, H, W = m.shape
        mask = torch.zeros(H, W, dtype=torch.bool, device=m.device)
        mask[:R, :] = True; mask[-R:, :] = True; mask[:, :R] = True; mask[:, -R:] = True
        ring = m[:, mask].mean().item()
        inner = m[:, ~mask].mean().item()
        acc['full'] += m.mean().item(); acc['ring'] += ring; acc['inner'] += inner
        acc['psnr'] += psnr(x, y); acc['n'] += 1
        acc['nring'] = int(mask.sum()); acc['ninner'] = int((~mask).sum())
    n = acc['n']
    frac_ring = acc['nring'] / (acc['nring'] + acc['ninner'])
    res[name] = dict(n=n, ssim_full=acc['full']/n, ssim_ring=acc['ring']/n,
                     ssim_inner=acc['inner']/n, psnr=acc['psnr']/n,
                     frac_ring=frac_ring,
                     ring_contrib=frac_ring*acc['ring']/n,
                     ssim_if_ring_perfect=(1-frac_ring)*acc['inner']/n + frac_ring*1.0)
    print(name, json.dumps(res[name], indent=1))
json.dump(res, open(os.path.dirname(__file__)+'/a1_border_diag.json','w'), indent=1)

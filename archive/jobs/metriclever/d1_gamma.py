"""(d) NEGATIVE CONTROL for the harness: global brightness offset + gamma FITTED ON TRAIN VIEWS,
applied to test renders.  Appearance is known-dead (oracle bound +0.09 dB) so this must read ~0.
Also reports the TEST-FITTED ORACLE for the same 2 params, to bound the axis honestly."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *

SETS = [('HCM0181_k4f', '/mnt/d/avv/prodharness/k4f', 'HCM0181',
         f'{OUT}/HCM0181_gsplatB9ut/train_renders'),
        ('HCM0193', f'{OUT}/HCM0193_gsplatB9ut/test_poses_renders_png', 'HCM0193',
         f'{OUT}/HCM0193_gsplatB9ut/train_renders'),
        ('HCM0204', f'{OUT}/HCM0204_gsplatB9ut/test_poses_renders_png', 'HCM0204',
         f'{OUT}/HCM0204_gsplatB9ut/train_renders'),
        ('hcm0031', f'{OUT}/hcm0031_gsplatB9ut/test_poses_renders_png', 'hcm0031',
         f'{OUT}/hcm0031_gsplatB9ut/train_renders'),
        ('hcm0034', f'{OUT}/hcm0034_gsplatB9ut/test_poses_renders_png', 'hcm0034',
         f'{OUT}/hcm0034_gsplatB9ut/train_renders')]

def apply(x, b, g):
    return ((x.clamp(1e-6, 1) ** g) + b).clamp(0, 1)

GG = [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]
BB = [-0.010, -0.005, -0.002, 0.0, 0.002, 0.005, 0.010]

out = {}
for name, rd, scene, trd in SETS:
    # --- fit on TRAIN views by minimising MSE (proxy the composite via psnr; appearance only) ---
    fit = None
    if os.path.isdir(trd):
        ts = [s for s in stems(trd) if os.path.exists(f'{PUB}/{scene}/train/images/{s}.JPG')]
        ts = ts[::max(1, len(ts)//25)][:25]
        best = (1e9, 0.0, 1.0)
        acc = {(b, g): 0. for b in BB for g in GG}
        for si in ts:
            x = load(f'{trd}/{si}.png'); y = load(f'{PUB}/{scene}/train/images/{si}.JPG')
            if x.shape != y.shape: continue
            for b in BB:
                for g in GG:
                    acc[(b, g)] += ((apply(x, b, g)-y)**2).mean().item()
        for k, v in acc.items():
            if v < best[0]: best = (v, k[0], k[1])
        fit = (best[1], best[2])
    # --- evaluate on TEST ---
    st = stems(rd)
    def ev(b, g):
        P = S = L = 0.
        for si in st:
            x = load(f'{rd}/{si}.png'); y = load(f'{PUB}/{scene}/test/images/{si}.JPG')
            z = apply(x, b, g)
            P += psnr(z, y); S += ssim_val(z, y); L += lpips_val(z, y)
        n = len(st)
        return score(P/n, S/n, L/n)
    base = ev(0.0, 1.0)
    trainfit = ev(*fit) if fit else None
    # test-fitted oracle over a small grid
    orc = (-1e9, None)
    for b in [-0.005, -0.002, 0.0, 0.002, 0.005]:
        for g in [0.96, 0.98, 1.0, 1.02, 1.04]:
            s = ev(b, g)
            if s > orc[0]: orc = (s, (b, g))
    out[name] = dict(base=base, train_fit_params=fit, train_fit_score=trainfit,
                     d_trainfit=(trainfit-base) if trainfit is not None else None,
                     oracle_score=orc[0], oracle_params=orc[1], d_oracle=orc[0]-base)
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(os.path.dirname(__file__)+'/d1_gamma.json', 'w'), indent=1)

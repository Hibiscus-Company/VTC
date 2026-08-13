"""(d) NEGATIVE CONTROL, I/O-efficient: load each image once, evaluate all (offset,gamma) configs.
Fit on TRAIN views, apply to TEST. Known-dead axis -> must read ~0 if the harness is honest.
Also reports the TEST-fitted oracle to bound the axis."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *

SC = ['HCM0181', 'HCM0193', 'HCM0204', 'hcm0031', 'hcm0034']
BB = [-0.010, -0.005, -0.002, 0.0, 0.002, 0.005, 0.010]
GG = [0.94, 0.97, 1.0, 1.03, 1.06]
CFG = [(b, g) for b in BB for g in GG]

def apply(x, b, g):
    return (x.clamp(1e-6, 1)**g + b).clamp(0, 1) if (b != 0.0 or g != 1.0) else x

out = {}
for scene in SC:
    trd = f'{OUT}/{scene}_gsplatB9ut/train_renders'
    rd = f'{OUT}/{scene}_gsplatB9ut/test_poses_renders_png'
    # ---- fit on TRAIN views: minimise MSE and also maximise composite (both reported) ----
    ts = [s for s in stems(trd) if os.path.exists(f'{PUB}/{scene}/train/images/{s}.JPG')][:30]
    tmse = {c: 0. for c in CFG}; tsc = {c: [0., 0., 0.] for c in CFG}
    for si in ts:
        x = load(f'{trd}/{si}.png'); y = load(f'{PUB}/{scene}/train/images/{si}.JPG')
        if x.shape != y.shape: continue
        for c in CFG:
            z = apply(x, *c)
            tmse[c] += ((z-y)**2).mean().item()
            a = tsc[c]; a[0] += psnr(z, y); a[1] += ssim_val(z, y); a[2] += lpips_val(z, y)
    nt = len(ts)
    fit_mse = min(CFG, key=lambda c: tmse[c])
    fit_sc = max(CFG, key=lambda c: score(*[tsc[c][i]/nt for i in range(3)]))
    # ---- evaluate every config on TEST ----
    st = stems(rd)
    acc = {c: [0., 0., 0.] for c in CFG}
    for si in st:
        x = load(f'{rd}/{si}.png'); y = load(f'{PUB}/{scene}/test/images/{si}.JPG')
        for c in CFG:
            z = apply(x, *c)
            a = acc[c]; a[0] += psnr(z, y); a[1] += ssim_val(z, y); a[2] += lpips_val(z, y)
    n = len(st)
    S = {c: score(*[acc[c][i]/n for i in range(3)]) for c in CFG}
    base = S[(0.0, 1.0)]
    orc = max(CFG, key=lambda c: S[c])
    out[scene] = dict(n_test=n, n_train=nt, base=base,
                      fit_mse=list(fit_mse), d_fit_mse=S[fit_mse]-base,
                      fit_score=list(fit_sc), d_fit_score=S[fit_sc]-base,
                      test_oracle=list(orc), d_test_oracle=S[orc]-base)
    print(f"{scene:9s} base={base:8.4f} | TRAIN-fit(MSE) b={fit_mse[0]:+.3f} g={fit_mse[1]:.2f} "
          f"-> d={S[fit_mse]-base:+.4f} | TRAIN-fit(score) b={fit_sc[0]:+.3f} g={fit_sc[1]:.2f} "
          f"-> d={S[fit_sc]-base:+.4f} | TEST-ORACLE b={orc[0]:+.3f} g={orc[1]:.2f} "
          f"-> d={S[orc]-base:+.4f}", flush=True)
m1 = sum(v['d_fit_mse'] for v in out.values())/len(out)
m2 = sum(v['d_fit_score'] for v in out.values())/len(out)
m3 = sum(v['d_test_oracle'] for v in out.values())/len(out)
print(f"\nMEAN over 5 scenes: train-fit(MSE) {m1:+.4f}   train-fit(score) {m2:+.4f}   test-oracle {m3:+.4f}")
out['_mean'] = dict(d_fit_mse=m1, d_fit_score=m2, d_test_oracle=m3)
json.dump(out, open(os.path.dirname(__file__)+'/d2_gamma.json', 'w'), indent=1)

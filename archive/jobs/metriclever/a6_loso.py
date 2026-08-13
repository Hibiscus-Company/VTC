"""Cross-scene validation of the local-contrast gain. 5 scenes x {plain, lens-field} renders
gives 10 (rho, dScore(g)) curves + the 2 HCM0181 ensembles. Leave-one-scene-out: fit the
gain rule on the other scenes, apply to the held-out scene, report the HONEST delta."""
import sys, os, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from utils.loss_utils import create_window

SC = ['HCM0181', 'HCM0193', 'HCM0204', 'hcm0031', 'hcm0034']
SETS = []
for s in SC:
    SETS.append((f'{s}|plain', f'{OUT}/{s}_gsplatB9ut/test_poses_renders_png', s))
    SETS.append((f'{s}|field', f'{OUT}/{s}_gsplatB9ut/test_poses_renders_field', s))
SETS.append(('HCM0181|k4', '/mnt/d/avv/prodharness/k4/png', 'HCM0181'))
SETS.append(('HCM0181|k4f', '/mnt/d/avv/prodharness/k4f', 'HCM0181'))
GAINS = [0.95, 1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15]
SIG = 1.0

def gk(sig, dev):
    r = max(1, int(round(3*sig))); k = torch.arange(-r, r+1, device=dev, dtype=torch.float32)
    g = torch.exp(-k*k/(2*sig*sig)); return g/g.sum(), r

def blur(x, sig):
    g, r = gk(sig, x.device)
    w1 = g[None, None, None, :].expand(3, 1, 1, 2*r+1)
    w2 = g[None, None, :, None].expand(3, 1, 2*r+1, 1)
    z = F.conv2d(F.pad(x[None], (r, r, 0, 0), mode='reflect'), w1, groups=3)
    return F.conv2d(F.pad(z, (0, 0, r, r), mode='reflect'), w2, groups=3)[0]

out = {}
for name, rd, scene in SETS:
    if not os.path.isdir(rd): print('skip', rd); continue
    ext = '.png'
    st = stems(rd, ext)
    if not st:
        st = stems(rd, '.jpg'); ext = '.jpg'
    acc = {g: [0., 0., 0.] for g in GAINS}
    ra = [0., 0., 0.]
    w = None
    for si in st:
        gp = f'{PUB}/{scene}/test/images/{si}.JPG'
        if not os.path.exists(gp): continue
        x0 = load(f'{rd}/{si}{ext}'); y = load(gp)
        if w is None: w = create_window(11, 3).type_as(x0)
        p = 5
        m1 = F.conv2d(x0[None], w, padding=p, groups=3); m2 = F.conv2d(y[None], w, padding=p, groups=3)
        ra[0] += (F.conv2d(x0[None]*x0[None], w, padding=p, groups=3)-m1*m1).mean().item()
        ra[1] += (F.conv2d(y[None]*y[None], w, padding=p, groups=3)-m2*m2).mean().item()
        ra[2] += (F.conv2d(x0[None]*y[None], w, padding=p, groups=3)-m1*m2).mean().item()
        mu = blur(x0, SIG); hp = x0-mu
        for g in GAINS:
            x = x0 if g == 1.0 else (mu + g*hp).clamp(0, 1)
            a = acc[g]
            a[0] += psnr(x, y); a[1] += ssim_val(x, y); a[2] += lpips_val(x, y)
    n = sum(1 for si in st if os.path.exists(f'{PUB}/{scene}/test/images/{si}.JPG'))
    b = score(*[acc[1.0][i]/n for i in range(3)])
    rho = ra[2]/((ra[0]*ra[1])**0.5)
    out[name] = dict(n=n, rho=rho, base=b,
                     d={str(g): score(*[acc[g][i]/n for i in range(3)])-b for g in GAINS})
    bg = max(GAINS, key=lambda g: out[name]['d'][str(g)])
    print(f"{name:16s} n={n:3d} rho={rho:.4f} base={b:8.4f} best g={bg:.3f} d={out[name]['d'][str(bg)]:+.4f}  "
          + ' '.join(f"{g}:{out[name]['d'][str(g)]:+.3f}" for g in GAINS), flush=True)
json.dump(out, open(os.path.dirname(__file__)+'/a6_loso.json', 'w'), indent=1)

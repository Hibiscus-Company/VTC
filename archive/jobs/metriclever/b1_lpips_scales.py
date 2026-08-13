"""(b) Which of LPIPS-vgg's 5 feature scales carries our deficit?
Decompose val = sum_k res[k] exactly (lpips sums 5 layer terms), per render variant.
Also: which scale delivered the KNOWN wins (k4 -> k4f lens field, single -> 4-member ensemble)
and which scale responds to blur/sharpen probes -> tells us the spatial band that matters."""
import sys, os, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lpips as lpips_pkg
from lpips import normalize_tensor, spatial_average

M = lpips_model()
LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
STRIDE = [1, 2, 4, 8, 16]

def lpips_perscale(x, y):
    with torch.no_grad():
        i0, i1 = M.scaling_layer(x[None]*2-1), M.scaling_layer(y[None]*2-1)
        o0, o1 = M.net.forward(i0), M.net.forward(i1)
        res = []
        for kk in range(M.L):
            f0, f1 = normalize_tensor(o0[kk]), normalize_tensor(o1[kk])
            d = (f0-f1)**2
            res.append(spatial_average(M.lins[kk].model(d), keepdim=True).item())
    return res

def gauss_blur(x, sig):
    r = max(1, int(3*sig)); k = torch.arange(-r, r+1, device=x.device, dtype=x.dtype)
    g = torch.exp(-k**2/(2*sig*sig)); g = g/g.sum()
    w = g[None, None, None, :].expand(3, 1, 1, 2*r+1)
    z = F.conv2d(F.pad(x[None], (r, r, 0, 0), mode='reflect'), w, groups=3)
    w2 = g[None, None, :, None].expand(3, 1, 2*r+1, 1)
    return F.conv2d(F.pad(z, (0, 0, r, r), mode='reflect'), w2, groups=3)[0]

VARIANTS = {
    'k4f (production-shaped)': ('/mnt/d/avv/prodharness/k4f', 'HCM0181'),
    'k4  (ensemble, no field)': ('/mnt/d/avv/prodharness/k4/png', 'HCM0181'),
    'B9ut (single member)':     (f'{OUT}/HCM0181_gsplatB9ut/test_poses_renders_png', 'HCM0181'),
}
out = {}
for name, (rd, scene) in VARIANTS.items():
    st = stems(rd); acc = [0.]*5; n = 0
    for si in st:
        x = load(f'{rd}/{si}.png'); y = load(f'{PUB}/{scene}/test/images/{si}.JPG')
        r = lpips_perscale(x, y)
        acc = [a+b for a, b in zip(acc, r)]; n += 1
    out[name] = dict(per_scale=[a/n for a in acc], total=sum(acc)/n, n=n)

# probes on k4f: blur / unsharp, to see which scale each spatial band drives
rd, scene = '/mnt/d/avv/prodharness/k4f', 'HCM0181'
st = stems(rd)[:20]
probes = {'blur s=0.5': lambda x: gauss_blur(x, 0.5),
          'blur s=1.0': lambda x: gauss_blur(x, 1.0),
          'unsharp 0.3 s=1': lambda x: (x + 0.3*(x-gauss_blur(x, 1.0))).clamp(0, 1),
          'gauss noise 2/255': lambda x: (x + torch.randn_like(x)*2/255).clamp(0, 1)}
pout = {}
base = [0.]*5
for si in st:
    x = load(f'{rd}/{si}.png'); y = load(f'{PUB}/{scene}/test/images/{si}.JPG')
    base = [a+b for a, b in zip(base, lpips_perscale(x, y))]
base = [a/len(st) for a in base]
for pn, fn in probes.items():
    acc = [0.]*5
    for si in st:
        x = load(f'{rd}/{si}.png'); y = load(f'{PUB}/{scene}/test/images/{si}.JPG')
        acc = [a+b for a, b in zip(acc, lpips_perscale(fn(x), y))]
    pout[pn] = [a/len(st)-b for a, b in zip(acc, base)]

print(f"{'variant':28s} " + ' '.join(f'{l:>9s}' for l in LAYERS) + f" {'TOTAL':>9s}")
for k, v in out.items():
    print(f"{k:28s} " + ' '.join(f'{a:9.5f}' for a in v['per_scale']) + f" {v['total']:9.5f}")
print()
print('deficit share (% of total LPIPS) per scale')
for k, v in out.items():
    print(f"{k:28s} " + ' '.join(f'{100*a/v[chr(116)+chr(111)+chr(116)+chr(97)+chr(108)]:8.2f}%' for a in v['per_scale']))
print()
print('dLPIPS per scale for known WIN  (k4 -> k4f lens field)')
d = [a-b for a, b in zip(out['k4f (production-shaped)']['per_scale'], out['k4  (ensemble, no field)']['per_scale'])]
print('  ' + ' '.join(f'{a:+9.5f}' for a in d), ' total', f'{sum(d):+.5f}')
d = [a-b for a, b in zip(out['k4  (ensemble, no field)']['per_scale'], out['B9ut (single member)']['per_scale'])]
print('dLPIPS per scale for known WIN  (single -> 4-member ensemble)')
print('  ' + ' '.join(f'{a:+9.5f}' for a in d), ' total', f'{sum(d):+.5f}')
print()
print(f"{'probe (n=20, on k4f)':28s} " + ' '.join(f'{l:>9s}' for l in LAYERS) + f" {'TOTAL':>9s}")
print(f"{'BASE k4f':28s} " + ' '.join(f'{a:9.5f}' for a in base) + f" {sum(base):9.5f}")
for k, v in pout.items():
    print(f"{k:28s} " + ' '.join(f'{a:+9.5f}' for a in v) + f" {sum(v):+9.5f}")
json.dump(dict(variants=out, probes=pout, base=base),
          open(os.path.dirname(__file__)+'/b1_lpips_scales.json', 'w'), indent=1)

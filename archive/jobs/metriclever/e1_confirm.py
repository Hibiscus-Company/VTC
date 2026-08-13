"""Confirm the best (a) config through the PRODUCTION ENCODE (JPEG q100, subsampling=2),
and (c) exhaustive PSNR-clamp check over every harness render set."""
import sys, os, io, json, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from common import *

def gk(sig, dev):
    r = max(1, int(round(3*sig))); k = torch.arange(-r, r+1, device=dev, dtype=torch.float32)
    g = torch.exp(-k*k/(2*sig*sig)); return g/g.sum(), r

def blur(x, sig):
    g, r = gk(sig, x.device)
    w1 = g[None, None, None, :].expand(3, 1, 1, 2*r+1)
    w2 = g[None, None, :, None].expand(3, 1, 2*r+1, 1)
    z = F.conv2d(F.pad(x[None], (r, r, 0, 0), mode='reflect'), w1, groups=3)
    return F.conv2d(F.pad(z, (0, 0, r, r), mode='reflect'), w2, groups=3)[0]

def jpeg_rt(x, q=100, ss=2):
    a = (x.clamp(0, 1)*255).round().byte().permute(1, 2, 0).cpu().numpy()
    b = io.BytesIO(); Image.fromarray(a).save(b, 'JPEG', quality=q, subsampling=ss)
    n = b.tell(); b.seek(0)
    return load_pil(Image.open(b).convert('RGB'), x.device), n

import numpy as np
def load_pil(im, dev):
    a = np.asarray(im, dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous().to(dev)

RD, SCENE = '/mnt/d/avv/prodharness/k4f', 'HCM0181'
GAINS = [1.0, 1.05, 1.10, 1.15]
SIG = 1.0
st = stems(RD)
acc = {g: [0., 0., 0., 0] for g in GAINS}
accp = {g: [0., 0., 0.] for g in GAINS}
for si in st:
    x0 = load(f'{RD}/{si}.png'); y = load(f'{PUB}/{SCENE}/test/images/{si}.JPG')
    mu = blur(x0, SIG); hp = x0-mu
    for g in GAINS:
        x = (mu + g*hp).clamp(0, 1) if g != 1.0 else x0
        accp[g][0] += psnr(x, y); accp[g][1] += ssim_val(x, y); accp[g][2] += lpips_val(x, y)
        xj, nb = jpeg_rt(x)
        a = acc[g]
        a[0] += psnr(xj, y); a[1] += ssim_val(xj, y); a[2] += lpips_val(xj, y); a[3] += nb
n = len(st)
print(f"{'gain':6s} {'PNG score':>10s} {'dPNG':>8s} | {'JPEG score':>10s} {'dJPEG':>8s} {'MB/scene':>9s}")
b_png = score(*[accp[1.0][i]/n for i in range(3)])
b_jpg = score(*[acc[1.0][i]/n for i in range(3)])
res = {}
for g in GAINS:
    sp = score(*[accp[g][i]/n for i in range(3)])
    sj = score(*[acc[g][i]/n for i in range(3)])
    mb = acc[g][3]/1e6
    res[g] = dict(png=sp, d_png=sp-b_png, jpeg=sj, d_jpeg=sj-b_jpg, mb=mb,
                  jpeg_psnr=acc[g][0]/n, jpeg_ssim=acc[g][1]/n, jpeg_lpips=acc[g][2]/n)
    print(f"{g:<6.2f} {sp:10.4f} {sp-b_png:+8.4f} | {sj:10.4f} {sj-b_jpg:+8.4f} {mb:9.1f}")

# (c) exhaustive clamp check
ALL = [('HCM0181_k4f', RD, 'HCM0181'), ('HCM0181_k4', '/mnt/d/avv/prodharness/k4/png', 'HCM0181')]
for sc_ in ['HCM0181', 'HCM0193', 'HCM0204', 'hcm0031', 'hcm0034']:
    d = f'{OUT}/{sc_}_gsplatB9ut/test_poses_renders_png'
    if os.path.isdir(d): ALL.append((sc_+'_B9ut', d, sc_))
clamp = {}
for nm, rd, sc_ in ALL:
    mx = -1e9; mn = 1e9
    for si in stems(rd):
        p = psnr(load(f'{rd}/{si}.png'), load(f'{PUB}/{sc_}/test/images/{si}.JPG'))
        mx = max(mx, p); mn = min(mn, p)
    clamp[nm] = dict(psnr_min=mn, psnr_max=mx, headroom_dB=50-mx)
    print(f"(c) {nm:16s} per-image PSNR min={mn:6.2f} max={mx:6.2f}  headroom to clamp={50-mx:6.2f} dB")
json.dump(dict(gain=res, clamp=clamp), open(os.path.dirname(__file__)+'/e1_confirm.json', 'w'), indent=1)

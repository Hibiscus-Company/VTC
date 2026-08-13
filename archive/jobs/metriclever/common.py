import sys, os, glob
sys.path.insert(0, '/mnt/c/Users/BKAI/an_plaza2/FastGS')
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from utils.loss_utils import ssim as repo_ssim, create_window

DEV = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
PUB = '/mnt/d/avv/data/phase1/public_set'
OUT = '/mnt/d/avv/output'

def load(p, dev=None):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).contiguous()
    return t.to(dev or DEV)

def stems(d, ext='.png'):
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(d + '/*' + ext))

def psnr(x, y):
    return (10.0 * torch.log10(1.0 / ((x - y) ** 2).mean())).item()

_lp = None
def lpips_model():
    global _lp
    if _lp is None:
        import lpips
        _lp = lpips.LPIPS(net='vgg').to(DEV).eval()
    return _lp

def lpips_val(x, y):
    m = lpips_model()
    with torch.no_grad():
        return m(x[None] * 2 - 1, y[None] * 2 - 1).item()

def ssim_map(x, y, ws=11):
    """Full ssim map, exactly the repo conv (zero pad, no window renorm)."""
    ch = x.size(-3)
    w = create_window(ws, ch).type_as(x).to(x.device)
    p = ws // 2
    mu1 = F.conv2d(x[None], w, padding=p, groups=ch)
    mu2 = F.conv2d(y[None], w, padding=p, groups=ch)
    m1s, m2s, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(x[None] * x[None], w, padding=p, groups=ch) - m1s
    s2 = F.conv2d(y[None] * y[None], w, padding=p, groups=ch) - m2s
    s12 = F.conv2d(x[None] * y[None], w, padding=p, groups=ch) - m12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * m12 + C1) * (2 * s12 + C2)) / ((m1s + m2s + C1) * (s1 + s2 + C2)))[0]

def ssim_val(x, y):
    return repo_ssim(x[None], y[None]).item()

def score(psnr_v, ssim_v, lpips_v):
    return 100 * (0.4 * (1 - lpips_v) + 0.3 * ssim_v + 0.3 * min(psnr_v, 50.0) / 50.0)

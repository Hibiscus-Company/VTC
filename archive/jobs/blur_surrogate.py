#!/usr/bin/env python
"""Is bonsai's error explained by pure under-resolution? Compare the real render's LPIPS
to the LPIPS of a BLURRED GT whose Laplacian variance matches the render's."""
import os, numpy as np, cv2, torch, lpips
torch.set_num_threads(12)
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"; RD = "/mnt/d/avv/bonsai_perc/pC_lpearly/eval_png"
def rd(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
def vl(g): return float(cv2.Laplacian(g.mean(2), cv2.CV_32F, ksize=3).var())
L = lpips.LPIPS(net='vgg')
T = lambda x: torch.from_numpy(x/255.).permute(2,0,1)[None].float()*2-1
names = sorted(os.listdir(GT))
res = []
for nm in names:
    g = rd(os.path.join(GT, nm)); r = rd(os.path.join(RD, nm.replace('.jpg','.png')))
    tgt = vl(r)
    lo, hi = 0.05, 6.0
    for _ in range(18):
        s = 0.5*(lo+hi); b = cv2.GaussianBlur(g, (0,0), s)
        if vl(b) > tgt: lo = s
        else: hi = s
    b = cv2.GaussianBlur(g, (0,0), s)
    with torch.no_grad():
        lr = float(L(T(g), T(r))); lb = float(L(T(g), T(b)))
    mse_r = max(float(((g-r)**2).mean()),1e-9); mse_b = max(float(((g-b)**2).mean()),1e-9)
    res.append((s, lr, lb, 10*np.log10(255**2/mse_r), 10*np.log10(255**2/mse_b)))
    print("%s sigma %.2f  LPIPS render %.4f  LPIPS blurGT %.4f  PSNR render %.2f blurGT %.2f" % ((nm,)+res[-1]), flush=True)
a = np.array(res)
print("MEAN sigma %.2f | LPIPS render %.4f blurGT %.4f (blur explains %.1f%%) | PSNR render %.2f blurGT %.2f" %
      (a[:,0].mean(), a[:,1].mean(), a[:,2].mean(), 100*a[:,2].mean()/a[:,1].mean(), a[:,3].mean(), a[:,4].mean()), flush=True)

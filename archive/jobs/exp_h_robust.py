"""(a-4) The border is variance-dominated. Does a ROBUST combiner beat the pixel-mean there?
Builds a 4-member HCM0181 ensemble from the ut family and compares mean / median / band-median."""
import sys
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from harness import *
import torch.nn.functional as Fn

MEM = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
DIRS = [f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png" for m in MEM]
GD = GT_TEST.format(s="HCM0181")


def dm(H, W):
    yy = torch.arange(H, device=DEV).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=DEV).view(1, W).expand(H, W)
    return torch.minimum(torch.minimum(yy, H - 1 - yy), torch.minimum(xx, W - 1 - xx))


def gkb(x, sig):
    ks = int(2 * round(3 * sig) + 1)
    t = torch.arange(ks, dtype=torch.float32, device=DEV) - ks // 2
    k = torch.exp(-t ** 2 / (2 * sig ** 2)); k /= k.sum(); C = x.shape[1]
    x = Fn.conv2d(Fn.pad(x, (ks // 2, ks // 2, 0, 0), mode="replicate"),
                  k.view(1, 1, 1, ks).expand(C, 1, 1, ks), groups=C)
    return Fn.conv2d(Fn.pad(x, (0, 0, ks // 2, ks // 2), mode="replicate"),
                     k.view(1, 1, ks, 1).expand(C, 1, ks, 1), groups=C)


stems = sorted(os.path.splitext(f)[0] for f in os.listdir(DIRS[0]) if f.endswith(".png"))
gt_by = {os.path.splitext(f)[0]: os.path.join(GD, f) for f in os.listdir(GD)}
D = None

# also: per-ring member DISAGREEMENT (is the border really higher-variance across members?)
print("member disagreement (std across the 4 members) by distance-from-border, first 20 imgs")
acc = torch.zeros(21, dtype=torch.float64, device=DEV); cnt = torch.zeros(21, dtype=torch.float64, device=DEV)
with torch.no_grad():
    for st in stems[:20]:
        M = torch.cat([load(os.path.join(d, st + ".png")).to(DEV) for d in DIRS], 0)
        H, W = M.shape[-2:]
        if D is None: Dc = dm(H, W).clamp(max=20).reshape(-1)
        sd = M.std(0, unbiased=True).mean(0).reshape(-1).double()
        acc.scatter_add_(0, Dc, sd); cnt.scatter_add_(0, Dc, torch.ones_like(sd))
        D = 1
p = (acc / cnt).cpu().numpy()
print("  d=0..9 :", " ".join(f"{p[i]:.5f}" for i in range(10)))
print("  d=10..19:", " ".join(f"{p[i]:.5f}" for i in range(10, 20)), " d>=20:", f"{p[20]:.5f}")

D = None
variants = ["mean", "median", "bandmed20", "bandmed40", "trimmean"]
res = {}
for v in variants:
    ps, ss, ls = [], [], []
    with torch.no_grad():
        for st in stems:
            M = torch.cat([load(os.path.join(d, st + ".png")).to(DEV) for d in DIRS], 0)
            g = load(gt_by[st]).to(DEV)
            H, W = M.shape[-2:]
            if D is None: D = dm(H, W)
            mean = M.mean(0, keepdim=True)
            if v == "mean":
                r = mean
            elif v == "median":
                r = M.median(0, keepdim=True).values
            elif v == "trimmean":
                srt = M.sort(0).values
                r = srt[1:3].mean(0, keepdim=True)
            else:
                bw = int(v[7:])
                w = gkb((D < bw).float().view(1, 1, H, W), max(bw / 3.0, 1.0))
                r = mean * (1 - w) + M.median(0, keepdim=True).values * w
            r = torch.round(r.clamp(0, 1) * 255) / 255
            ps.append(10 * np.log10(1 / max(((r - g) ** 2).mean().item(), 1e-12)))
            ss.append(float(repo_ssim(r, g)))
            ls.append(float(getvgg()(r * 2 - 1, g * 2 - 1).item()))
    P, S, L = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
    res[v] = score(P, S, L)
    print(f"{v:12s} PSNR {P:8.4f} SSIM {S:.5f} LPIPS {L:.5f} SCORE {res[v]:8.4f} "
          f"dScore {res[v]-res['mean']:+.4f}", flush=True)

"""Low-GPU-memory variant: renders live in CPU RAM, per-image slices streamed to GPU."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(TMP, "lever_ens")
NAMES = open(os.path.join(TMP, "names.txt")).read().split("\n")
IDX = {n: i for i, n in enumerate(NAMES)}
_R = _G = _LP = None
DEV = "cuda:0"


def init(device="cuda:0"):
    global _R, _G, _LP, DEV
    DEV = device
    _R = np.load(os.path.join(TMP, "renders_u8.npy"), mmap_mode="r")
    _R = np.ascontiguousarray(_R)
    G = np.load(os.path.join(TMP, "gt_u8.npy"))
    _G = torch.from_numpy(G).to(DEV).permute(0, 3, 1, 2).float().contiguous() / 255.0
    _LP = lpips_pkg.LPIPS(net="vgg", verbose=False).to(DEV).eval()
    for p in _LP.parameters():
        p.requires_grad_(False)
    return _R.shape


def resolve(members):
    return [IDX[m] if isinstance(m, str) else m for m in members]


def build(members, gi, weights=None, round_u8=True):
    j = resolve(members)
    stack = torch.from_numpy(_R[j, gi]).to(DEV, non_blocking=True).float()
    if weights is None:
        out = stack.mean(0)
    else:
        w = torch.as_tensor(weights, device=DEV, dtype=torch.float32).view(-1, 1, 1, 1)
        out = (stack * w).sum(0) / w.sum()
    out = torch.clamp(out + 0.5, 0, 255).floor() if round_u8 else torch.clamp(out, 0, 255)
    return (out.permute(2, 0, 1).unsqueeze(0) / 255.0).contiguous()


@torch.no_grad()
def metrics_img(img, gi):
    gt = _G[gi:gi + 1]
    mse = ((img - gt) ** 2).reshape(1, -1).mean(1)
    return ((20 * torch.log10(1.0 / torch.sqrt(mse))).item(),
            repo_ssim(img, gt).item(),
            _LP(img * 2 - 1, gt * 2 - 1).item())


def summarize(P, S, L):
    P, S, L = np.asarray(P), np.asarray(S), np.asarray(L)
    d = dict(psnr=float(P.mean()), ssim=float(S.mean()), lpips=float(L.mean()))
    d["score"] = 100 * (0.4 * (1 - d["lpips"]) + 0.3 * d["ssim"] + 0.3 * min(d["psnr"] / 50, 1.0))
    return d


@torch.no_grad()
def score(members, weights=None, images=None, round_u8=True, per_image=False):
    if images is None:
        images = list(range(60))
    P, S, L = [], [], []
    for gi in images:
        p, s, l = metrics_img(build(members, gi, weights, round_u8), gi)
        P.append(p); S.append(s); L.append(l)
    d = summarize(P, S, L)
    if per_image:
        d["per"] = dict(psnr=P, ssim=S, lpips=L, imgs=list(images))
    return d


def score_subset(per, images):
    pos = {g: k for k, g in enumerate(per["imgs"])}
    k = [pos[g] for g in images]
    return summarize(np.array(per["psnr"])[k], np.array(per["ssim"])[k], np.array(per["lpips"])[k])

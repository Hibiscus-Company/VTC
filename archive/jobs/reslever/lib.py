import os, sys, json
import numpy as np, cv2, torch
sys.path.insert(0, "/mnt/d/avv/metric_probe")
import mlib

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = TMP + "/reslever"
DEV = "cuda:1"
NAMES = open(TMP + "/names.txt").read().split("\n")
IDX = {n: i for i, n in enumerate(NAMES)}
MEM = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
SCENES = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
GTD = "/mnt/d/avv/data/phase1/public_set/{s}/test/images"


def cache():
    R = np.load(TMP + "/renders_u8.npy", mmap_mode="r")
    G = np.load(TMP + "/gt_u8.npy", mmap_mode="r")
    return R, G


def apply_field(img_f32, field, scale=1.0):
    """img_f32: HxWx3 float [0,1]."""
    H, W, _ = img_f32.shape
    fu = cv2.resize(field * scale, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.clip(cv2.remap(img_f32,
                             (xx + fu[..., 0]).astype(np.float32),
                             (yy + fu[..., 1]).astype(np.float32),
                             cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), 0, 1)


_lp = None


def lp():
    global _lp
    if _lp is None:
        _lp = mlib.LP(device=DEV, net="vgg")
    return _lp


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1).unsqueeze(0).to(DEV)


@torch.no_grad()
def score(pred, gt):
    P, G = t(pred), t(gt)
    mse = float(((P - G) ** 2).mean())
    return (10 * np.log10(1.0 / max(mse, 1e-12)), float(mlib.ssim(P, G)), lp()(P, G))


def comp(p, s, l):
    return 100 * (0.4 * (1 - l) + 0.3 * s + 0.3 * min(p / 50.0, 1.0))


def scene_stack(scene, variant="gsplatB9ut"):
    """returns (renders float32 [N,H,W,3], gt float32) for non-181 scenes."""
    from PIL import Image
    d = GTD.format(s=scene)
    gts = sorted(os.listdir(d))
    rd = f"/mnt/d/avv/output/{scene}_{variant}/test_poses_renders_png"
    rmap = {os.path.splitext(f)[0]: f for f in os.listdir(rd)}
    Rs, Gs = [], []
    for f in gts:
        st = os.path.splitext(f)[0]
        Gs.append(np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), np.uint8))
        Rs.append(np.asarray(Image.open(os.path.join(rd, rmap[st])).convert("RGB"), np.uint8))
    return np.stack(Rs), np.stack(Gs), [os.path.splitext(f)[0] for f in gts]

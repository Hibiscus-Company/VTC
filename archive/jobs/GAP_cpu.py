"""TRAIN-vs-EVAL GAP, same model both sides, CPU only (GPUs are busy with live sweep arms).
PSNR + SSIM (repo_ssim, identical to scripts/eval_score.py). LPIPS deliberately NOT computed
here -- it is the GPU-hungry part; the overfitting signature is readable in PSNR/SSIM.
Prints per-image arrays so the gap gets a real CI.
"""
import os, sys, json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
from utils.loss_utils import ssim as repo_ssim

torch.set_num_threads(os.cpu_count() or 8)


def load(p):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)


def run(render_dir, gt_dir, tag, limit=None):
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)}
    rends = [f for f in sorted(os.listdir(render_dir))
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")]
    rends = [f for f in rends if os.path.splitext(f)[0] in gt_by]
    if limit:
        idx = np.linspace(0, len(rends) - 1, limit).round().astype(int)
        rends = [rends[i] for i in sorted(set(idx))]
    ps, ss = [], []
    with torch.no_grad():
        for f in rends:
            s = os.path.splitext(f)[0]
            r = load(os.path.join(render_dir, f))
            g = load(os.path.join(gt_dir, gt_by[s]))
            assert r.shape == g.shape, f"{f} {r.shape} {g.shape}"
            mse = ((r - g) ** 2).mean().item()
            ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
            ss.append(float(repo_ssim(r, g)))
    ps, ss = np.array(ps), np.array(ss)
    print(f"{tag:28s} n={len(ps):3d}  PSNR {ps.mean():7.4f} (sd {ps.std(ddof=1):.3f}, "
          f"sem {ps.std(ddof=1)/np.sqrt(len(ps)):.3f})  SSIM {ss.mean():.4f} "
          f"(sd {ss.std(ddof=1):.4f}, sem {ss.std(ddof=1)/np.sqrt(len(ss)):.4f})", flush=True)
    return dict(tag=tag, n=len(ps), psnr=ps.tolist(), ssim=ss.tolist())


ES = "/mnt/d/avv/evalsplit"
BB = "/mnt/d/avv/blurbound"
TW = "/mnt/d/avv/tw_test"
JOBS = [
    # scene, model, train_render, train_gt, eval_render, eval_gt
    ("bonsai", f"{BB}/bonsai/train_png", f"{ES}/bonsai/train_sub/images",
     f"{TW}/bonsai_ema099/eval_png", f"{ES}/bonsai/eval_gt"),
    ("chair", f"{BB}/chair/train_png", f"{ES}/chair/train_sub/images",
     f"{TW}/chair_ema099/eval_png", f"{ES}/chair/eval_gt"),
]
LIM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out = {}
for scene, tr, trg, ev, evg in JOBS:
    out[f"{scene}_TRAIN"] = run(tr, trg, f"{scene} TRAIN (in-sample)", LIM or None)
    out[f"{scene}_EVAL"] = run(ev, evg, f"{scene} EVAL (held-out)", None)
    a, b = out[f"{scene}_TRAIN"], out[f"{scene}_EVAL"]
    dp = np.mean(a["psnr"]) - np.mean(b["psnr"])
    ds = np.mean(a["ssim"]) - np.mean(b["ssim"])
    sep = np.sqrt(np.var(a["psnr"], ddof=1) / a["n"] + np.var(b["psnr"], ddof=1) / b["n"])
    ses = np.sqrt(np.var(a["ssim"], ddof=1) / a["n"] + np.var(b["ssim"], ddof=1) / b["n"])
    print(f"  >>> {scene} GAP  dPSNR {dp:+.4f} +/- {sep:.4f}   dSSIM {ds:+.5f} +/- {ses:.5f}\n",
          flush=True)
json.dump(out, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/GAP_cpu.json", "w"))
print("wrote GAP_cpu.json")

import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens")
import harn
harn.init()
R = harn.R(); G = harn.G()
idx = torch.as_tensor([1, 2, 3, 4, 5, 6], device="cuda:0")


def tt(fn, n=60):
    torch.cuda.synchronize(); t0 = time.time()
    for gi in range(n):
        fn(gi)
    torch.cuda.synchronize(); return time.time() - t0


def bld(gi):
    stack = R[idx, gi].float()
    return (torch.clamp(stack.mean(0) + 0.5, 0, 255).floor().permute(2, 0, 1).unsqueeze(0) / 255.0).contiguous()


print("build     ", tt(lambda gi: bld(gi)))
print("+psnr     ", tt(lambda gi: ((bld(gi) - G[gi:gi+1])**2).reshape(1, -1).mean(1)))
print("+ssim     ", tt(lambda gi: harn.repo_ssim(bld(gi), G[gi:gi+1])))
print("+lpips    ", tt(lambda gi: harn._LP(bld(gi)*2-1, G[gi:gi+1]*2-1)))
print("full      ", tt(lambda gi: harn.metrics_img(bld(gi), gi)))

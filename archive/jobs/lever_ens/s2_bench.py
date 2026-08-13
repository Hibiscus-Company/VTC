import os, sys, time, itertools, numpy as np, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens")
import harn
harn.init()
R = harn.R()
N = harn.NAMES

# --- identify k4 composition by brute force over 4-subsets of the 21 model variants ---
models = [i for i, n in enumerate(N) if n != "k4"]
k4i = N.index("k4")
best = None
gi = 0
target = R[k4i, gi].float()
for combo in itertools.combinations(models, 4):
    idx = torch.as_tensor(combo, device="cuda:0")
    m = torch.clamp(R[idx, gi].float().mean(0) + 0.5, 0, 255).floor()
    d = (m - target).abs().mean().item()
    if best is None or d < best[0]:
        best = (d, combo)
print("k4 best-match 4-subset:", [N[i] for i in best[1]], "mean|diff|", best[0], flush=True)

# --- benchmark batching ---
for B in (1, 2, 4, 6):
    torch.cuda.synchronize(); t0 = time.time()
    P = []
    for s in range(0, 60, B):
        imgs = list(range(s, min(s + B, 60)))
        idx = torch.as_tensor([1, 2, 3, 4], device="cuda:0")
        stack = R[idx][:, imgs].float()
        out = torch.clamp(stack.mean(0) + 0.5, 0, 255).floor().permute(0, 3, 1, 2) / 255.0
        gt = harn.G()[imgs]
        with torch.no_grad():
            l = harn._LP(out * 2 - 1, gt * 2 - 1)
        P.append(l.mean().item())
    torch.cuda.synchronize()
    print(f"batch {B}: {time.time()-t0:.2f}s  lpips {np.mean(P):.5f}  mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

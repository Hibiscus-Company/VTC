"""EXP-4  ALIGN-THEN-AVERAGE  (burst-photography stacking applied to the ensemble).

Pixel-mean averaging is only variance-reducing if the members are REGISTERED to each other.
If member m carries its own sub-pixel registration error, the mean is a blur of misaligned
copies -- exactly the HF deficit the team has been trying to patch downstream with texture
injection and sharpening.  Fixing it at the source is strictly better than patching it.

Crucially this uses NO GT: members are aligned to their own running mean, so it is directly
applicable at test time on the private set.

  plain      mean_m  X_m                       (production today)
  aligned    mean_m  shift(X_m, d_m)           d_m fit member->mean, GT-free, 2 rounds
"""
import numpy as np, torch, torch.nn.functional as F, json, hlib

DEV = 'cuda'


def shift_img(t, dx, dy):
    _, _, H, W = t.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=t.device, dtype=torch.float32),
                            torch.arange(W, device=t.device, dtype=torch.float32),
                            indexing='ij')
    grid = torch.stack([((xx + dx) / (W - 1)) * 2 - 1,
                        ((yy + dy) / (H - 1)) * 2 - 1], -1).unsqueeze(0)
    return F.grid_sample(t, grid, mode='bicubic', padding_mode='border', align_corners=True)


def fit_shift(r, g, iters=4):
    dx = dy = 0.0
    gxk = torch.tensor([[[[-.5, 0, .5]]]], device=r.device).repeat(3, 1, 1, 1)
    gyk = gxk.transpose(-1, -2)
    for _ in range(iters):
        w = shift_img(r, dx, dy)
        Ix = F.conv2d(F.pad(w, (1, 1, 0, 0), mode='replicate'), gxk, groups=3)
        Iy = F.conv2d(F.pad(w, (0, 0, 1, 1), mode='replicate'), gyk, groups=3)
        e = g - w
        m = torch.zeros_like(e[:, :1]); m[..., 8:-8, 8:-8] = 1
        Ix, Iy, e = Ix * m, Iy * m, e * m
        A = torch.tensor([[(Ix * Ix).sum(), (Ix * Iy).sum()],
                          [(Ix * Iy).sum(), (Iy * Iy).sum()]], dtype=torch.float64)
        b = torch.tensor([(Ix * e).sum(), (Iy * e).sum()], dtype=torch.float64)
        s = torch.linalg.solve(A, b)
        dx += float(s[0]); dy += float(s[1])
        if abs(float(s[0])) < 1e-4 and abs(float(s[1])) < 1e-4:
            break
    return dx, dy


hlib.init()
N = hlib.names(); R = hlib.renders(); GU8 = np.asarray(hlib.gt())
SETS = {
    'k4': ['gsplatB9ut', 'gsplatB10ut8M', 'gsplatB11ut60k', 'gsplatB12ut8Ms7'],
    'k6': ['gsplatB9ut', 'gsplatB10ut8M', 'gsplatB11ut60k', 'gsplatB12ut8Ms7',
           'gsplatB8pure', 'e17visnorm'],
}
n = len(GU8)
out = {}
for setname, MEM in SETS.items():
    idx = [N.index(m) for m in MEM]
    plain = np.zeros(GU8.shape, np.float32)
    for i in idx:
        plain += R[i].astype(np.float32) / 255.0
    plain /= len(idx)

    aligned = np.zeros_like(plain)
    allsh = []
    for v in range(n):
        ms = [torch.from_numpy(R[i][v].astype(np.float32) / 255.0
                               ).permute(2, 0, 1).unsqueeze(0).to(DEV) for i in idx]
        ref = torch.stack(ms).mean(0)
        cur = ms
        for _ in range(2):                       # 2 rounds: align to mean, recompute mean
            sh = [fit_shift(m, ref) for m in ms]
            cur = [shift_img(m, d[0], d[1]) for m, d in zip(ms, sh)]
            ref = torch.stack(cur).mean(0)
        allsh.append(sh)
        aligned[v] = ref[0].permute(1, 2, 0).cpu().numpy()
    allsh = np.array(allsh)                                     # (views, members, 2)
    # spread of member shifts WITHIN a view = the misregistration averaging is blurring over
    spread = allsh.std(axis=1).mean(0)
    print(f"\n[{setname}] inter-member shift spread within a view: "
          f"dx {spread[0]:.4f} px, dy {spread[1]:.4f} px   "
          f"(per-member bias dx {allsh[:,:,0].mean(0).round(3)})")
    out[setname + '_plain'] = hlib.score(plain, GU8, f"[{setname}] plain mean")
    out[setname + '_align'] = hlib.score(aligned, GU8, f"[{setname}] ALIGNED mean")
    d = out[setname + '_align']['score'] - out[setname + '_plain']['score']
    print(f"   dScore {d:+.4f}")

print("\n%-34s %8s %9s" % ("config", "score", "dScore"))
for s in SETS:
    b = out[s + '_plain']['score']
    for k in ('plain', 'align'):
        v = out[s + '_' + k]
        print("%-34s %8.4f  %+8.4f" % (v['tag'], v['score'], v['score'] - b))
json.dump(out, open('/home/bkai/.claude/jobs/1c9cf7e9/tmp/tj/exp4.json', 'w'), indent=1)

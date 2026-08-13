#!/usr/bin/env python
"""LEVER: multi-scale (Laplacian) fusion of ensemble members -- decisive re-measurement + the
controls that were missing.

The winning rule from the earlier sweep is 'energy' at the FINEST pyramid level only:
    out = pixmean + lam * (r - 1) * L0(pixmean)      r = sqrt(mean_i E_i / E(pixmean))
i.e. a SPATIALLY-VARYING level-0 unsharp whose gain map is the ensemble-disagreement ratio.
Everything else (median, max-magnitude, p-norm, injection, deeper levels) already lost.

What this script adds:
  S0  independent re-measure of the headline (fresh compute, no cache reuse)
  S1  THE MISSING CONTROL: is the gain from ENSEMBLE DISAGREEMENT or just from IMAGE STRUCTURE?
      'struct' = the identical boost histogram, re-ordered by the local energy of the MEAN alone
                 (no ensemble information whatsoever).  If struct ~= raw, this is not fusion, it
                 is adaptive sharpening and the whole 'members disagree at HF' story is wrong.
      'split'  = disagreement map estimated from members {1,2} only, APPLIED to the mean of all 4.
                 Tests that the map is a stable scene property, not a fit to these members.
  S2  k-curve to k=6 (production ships 6+1 members; a cleanup-class trick DECAYS with depth)
  S3  cross-scene: HCM0421 (different tower) lambda sweep -- does the fitted lam transfer?
  S4  full shipped chain: fuse -> median lens field -> JPEG q100 ss2
"""
import os, sys, io, json, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
from lapfuse import lap_pyr, lap_recon, fuse_image, _K, boxf
Image.MAX_IMAGE_PIXELS = None

DEV = "cuda"
GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
ROOT = "/mnt/d/avv/output"
JPEG_KW = dict(quality=100, subsampling=2, optimize=True, progressive=True)
OUT = os.path.join(HERE, "lv_rows")
os.makedirs(OUT, exist_ok=True)

from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
VGG = lpips_pkg.LPIPS(net="vgg").to(DEV).eval()
K = _K.to(DEV)


def score(P, S, L):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))


def agg(rows):
    P, S, L = rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 2].mean()
    return score(P, S, L), P, S, L


def dscore(rows, base):
    """exact per-view score deltas (the score is linear in P,S,L up to the PSNR clamp)"""
    f = lambda r: 100 * (0.4 * (1 - r[:, 2]) + 0.3 * r[:, 1] + 0.3 * r[:, 0] / 50.0)
    return f(rows) - f(base)


# ------------------------------------------------------------------ fusion (level-0 energy rule)
def energy_map(Ls, win, rmax=4.0, sub=None):
    """r map. Ls [k,3,H,W] level-0 laplacians. sub = member subset used to ESTIMATE disagreement."""
    mean = Ls.mean(0, keepdim=True)
    src = Ls if sub is None else Ls[sub]
    Em = boxf((src ** 2).sum(1, keepdim=True), win).mean(0, keepdim=True)
    Eb = boxf((mean ** 2).sum(1, keepdim=True), win)
    return torch.sqrt((Em + 1e-10) / (Eb + 1e-10)).clamp(max=rmax), mean, Eb


def hist_match(target_vals, order_by):
    """return a map with target_vals' HISTOGRAM, assigned in the rank order of order_by."""
    flat_t = torch.sort(target_vals.reshape(-1))[0]
    idx = torch.argsort(order_by.reshape(-1))
    out = torch.empty_like(flat_t)
    out[idx] = flat_t
    return out.reshape(order_by.shape)


def fuse_energy(Ls_stack, lam, win, mode="raw", sub=None, nlev=5):
    """full pyramid fuse with the energy rule at level 0 only."""
    laps, res, sizes = lap_pyr(Ls_stack, nlev, K)
    L0 = laps[0]
    r, mean0, Eb = energy_map(L0, win, sub=sub)
    if mode == "struct":                       # same histogram, ordered by IMAGE STRUCTURE only
        r = hist_match(r, Eb)
    elif mode == "const":
        r = r.mean().expand_as(r)
    elif mode == "shuffle":
        flat = r.reshape(-1)
        r = flat[torch.randperm(flat.numel(), device=r.device)].reshape(r.shape)
    out = [mean0 * (1.0 + lam * (r - 1.0))] + [l.mean(0, keepdim=True) for l in laps[1:]]
    return lap_recon(out, res.mean(0, keepdim=True), sizes, K)


# ------------------------------------------------------------------ harness
class H:
    def __init__(self, dirs, gtdir=GT, names=None):
        gtf = {os.path.splitext(f)[0]: f for f in os.listdir(gtdir)}
        self.stems = sorted(s for s in gtf
                            if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
        self.M = [np.stack([np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"))
                            for d in dirs]) for s in self.stems]
        self.G = [np.asarray(Image.open(os.path.join(gtdir, gtf[s])).convert("RGB"))
                  for s in self.stems]
        self.names = names or dirs
        print(f"  harness: {len(self.stems)} views x {len(dirs)} members {self.names}", flush=True)

    def stack(self, i, sub=None):
        a = self.M[i] if sub is None else self.M[i][sub]
        return torch.from_numpy(a).to(DEV).float().permute(0, 3, 1, 2) / 255.0

    def gtt(self, i):
        return torch.from_numpy(self.G[i]).to(DEV).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    def eval_img(self, a_u8, i):
        r = a_u8.float().to(DEV) / 255.0
        if r.dim() == 3:
            r = r.permute(2, 0, 1).unsqueeze(0)
        g = self.gtt(i)
        with torch.no_grad():
            mse = ((r - g) ** 2).mean().item()
            return (10 * np.log10(1.0 / max(mse, 1e-12)), float(repo_ssim(r, g)),
                    float(VGG(r * 2 - 1, g * 2 - 1).item()))

    def run(self, lam, win=3, mode="raw", sub=None, jpeg=False, members=None, field=None, tag=""):
        cache = os.path.join(OUT, f"{tag}.npy")
        if tag and os.path.exists(cache):
            return np.load(cache)
        rows = []
        for i in range(len(self.stems)):
            st = self.stack(i, members)
            img = st.mean(0, keepdim=True) if lam == 0 else fuse_energy(st, lam, win, mode, sub)
            a = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
            if field is not None:
                a = (np.clip(apply_field(a.astype(np.float32) / 255.0, field), 0, 1) * 255
                     ).round().astype(np.uint8)
            if jpeg:
                buf = io.BytesIO(); Image.fromarray(a).save(buf, "JPEG", **JPEG_KW); buf.seek(0)
                a = np.asarray(Image.open(buf).convert("RGB"))
            rows.append(self.eval_img(torch.from_numpy(np.ascontiguousarray(a)), i))
            del st, img
        rows = np.array(rows)
        if tag:
            np.save(cache, rows)
        return rows


def tbl(name, rows, base, extra=""):
    s, P, S, L = agg(rows)
    d = dscore(rows, base)
    print(f"{name:<34} {s:9.4f} {s - agg(base)[0]:+8.4f} {P:8.4f} {S:7.4f} {L:8.4f} "
          f"{(d > 0).sum():>3d}/{len(d)} {extra}", flush=True)
    return s - agg(base)[0], d


HDR = (f"{'config':<34} {'SCORE':>9} {'dScore':>8} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'win':>7}")

UT = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
D = lambda t: f"{ROOT}/HCM0181_{t}/test_poses_renders_png"

if __name__ == "__main__":
    torch.manual_seed(0)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    # ================================================================ S0 headline reproduce
    if stage in ("all", "s0"):
        print("\n" + "=" * 96)
        print("S0. HEADLINE, k=4 production members, fresh compute. PNG then shipped JPEG q100/ss2.")
        print("=" * 96)
        h = H([D(t) for t in UT], names=UT)
        b_png = h.run(0, tag="k4_png_base")
        b_jpg = h.run(0, jpeg=True, tag="k4_jpg_base")
        print(HDR)
        tbl("pixel-mean  [PNG] BASELINE", b_png, b_png)
        for lam in (0.5, 0.75, 1.0, 1.25, 1.5):
            tbl(f"energy lam{lam} win3  [PNG]", h.run(lam, tag=f"k4_png_l{lam}"), b_png)
        print()
        tbl("pixel-mean  [JPEG] BASELINE", b_jpg, b_jpg)
        for lam in (0.5, 0.75, 1.0, 1.25, 1.5):
            tbl(f"energy lam{lam} win3  [JPEG]", h.run(lam, jpeg=True, tag=f"k4_jpg_l{lam}"), b_jpg)
        del h; torch.cuda.empty_cache()

    # ================================================================ S1 THE MISSING CONTROL
    if stage in ("all", "s1"):
        print("\n" + "=" * 96)
        print("S1. IS THE ENSEMBLE-DISAGREEMENT MAP LOAD-BEARING, or is this adaptive sharpening?")
        print("    struct  = IDENTICAL boost histogram, re-ordered by the local energy of the MEAN")
        print("              alone -- contains ZERO ensemble information.")
        print("    split   = map estimated from members {1,2} only, applied to the mean of all 4.")
        print("    const/shuffle = same average boost / same histogram scrambled.")
        print("=" * 96)
        h = H([D(t) for t in UT], names=UT)
        b_png = h.run(0, tag="k4_png_base")
        print(HDR)
        tbl("pixel-mean BASELINE", b_png, b_png)
        for mode in ("raw", "struct", "const", "shuffle"):
            tbl(f"energy lam1.0 map={mode}", h.run(1.0, mode=mode, tag=f"k4_png_m{mode}"), b_png)
        tbl("energy lam1.0 split{1,2}->all4",
            h.run(1.0, sub=[0, 1], tag="k4_png_split01"), b_png)
        tbl("energy lam1.0 split{3,4}->all4",
            h.run(1.0, sub=[2, 3], tag="k4_png_split23"), b_png)
        # how correlated are the two independent halves' maps, and r vs structure?
        st = h.stack(0)
        L0 = lap_pyr(st, 5, K)[0][0]
        rA = energy_map(L0, 3, sub=[0, 1])[0].reshape(-1)
        rB = energy_map(L0, 3, sub=[2, 3])[0].reshape(-1)
        rAll, _, Eb = energy_map(L0, 3)
        c = lambda a, b: float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))
        print(f"\n    map corr r{{1,2}} vs r{{3,4}}            = {c(rA, rB):.4f}")
        print(f"    map corr r(all) vs log E_mean(struct) = {c(rAll.reshape(-1), torch.log(Eb.reshape(-1) + 1e-10)):.4f}")
        del h; torch.cuda.empty_cache()

    # ================================================================ S2 k-curve
    if stage in ("all", "s2"):
        print("\n" + "=" * 96)
        print("S2. k-CURVE. Production towers ship 6 members + 1 mip3d. A cleanup-class trick DECAYS")
        print("    with ensemble depth (that is how EMA and the JPEG 'fix' betrayed us); a genuine")
        print("    texture restoration should GROW, because deeper means = flatter HF.")
        print("=" * 96)
        EXTRA = ["e15ceil95", "e17visnorm"]
        allm = UT + EXTRA
        h = H([D(t) for t in allm], names=allm)
        print(f"{'k':>2}  {'pixel-mean':>10}  {'lam1.0':>10}  {'d':>9}  {'lam0.75':>10}  {'d':>9}")
        for k in (2, 3, 4, 5, 6):
            sub = list(range(k))
            b = h.run(0, members=sub, tag=f"kc{k}_base")
            e1 = h.run(1.0, members=sub, tag=f"kc{k}_l1.0")
            e075 = h.run(0.75, members=sub, tag=f"kc{k}_l0.75")
            sb, s1, s7 = agg(b)[0], agg(e1)[0], agg(e075)[0]
            print(f"{k:>2}  {sb:10.4f}  {s1:10.4f}  {s1 - sb:+9.4f}  {s7:10.4f}  {s7 - sb:+9.4f}",
                  flush=True)
        del h; torch.cuda.empty_cache()

    # ================================================================ S3 cross-scene
    if stage in ("all", "s3"):
        print("\n" + "=" * 96)
        print("S3. CROSS-SCENE. HCM0181 is the only scene with >1 render variant at REAL test poses,")
        print("    so the only other real multi-member novel-view data is the HCM0421 eval-split")
        print("    (DIFFERENT tower, held-out real photos as GT, k=2: base UT + mip3d -- the exact")
        print("    two families production blends). Data-starved regime: read the SIGN and the")
        print("    location of the optimum, not the magnitude.")
        print("=" * 96)
        G4 = "/mnt/d/avv/evalsplit/HCM0421/eval_gt"
        M4 = ["/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/mip3d/HCM0421_mip0.2/eval_png"]
        h = H(M4, gtdir=G4, names=["UT", "mip3d"])
        b = h.run(0, tag="x421_base")
        print(HDR)
        tbl("HCM0421 pixel-mean k=2", b, b)
        for lam in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            tbl(f"HCM0421 energy lam{lam} win3", h.run(lam, tag=f"x421_l{lam}"), b)
        del h; torch.cuda.empty_cache()

    # ================================================================ S4 shipped chain
    if stage in ("all", "s4"):
        print("\n" + "=" * 96)
        print("S4. FULL SHIPPED CHAIN: fuse -> median lens field (cv2.remap) -> JPEG q100 ss2.")
        print("    The field is a sub-pixel resample, exactly the operator that could eat a")
        print("    finest-level texture boost. Field fit on HCM0181 TRAIN photos (no test leakage).")
        print("=" * 96)
        sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
        from fit_field import apply_field as _af
        globals()["apply_field"] = _af
        FIELD = np.load(os.path.join(HERE, "HCM0181_median.npy"))
        h = H([D(t) for t in UT], names=UT)
        b = h.run(0, field=FIELD, jpeg=True, tag="chain_base")
        print(HDR)
        tbl("pixmean +field +JPEG (SHIPPED)", b, b)
        for lam in (0.5, 0.75, 1.0, 1.25):
            tbl(f"energy lam{lam} +field +JPEG",
                h.run(lam, field=FIELD, jpeg=True, tag=f"chain_l{lam}"), b)
        del h; torch.cuda.empty_cache()
    print("\nDONE", time.strftime("%H:%M:%S"))

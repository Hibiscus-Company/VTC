"""AUDIT PHASE 1 -- pure numerics, no GT needed.
(a) is lap_recon an exact inverse of lap_pyr at nlev=5?
(b) does nlev matter at all for restore()?
(c) what are the true shipped ensemble weights / k for the private towers?
(d) r-map statistics on the REAL shipped production data: clamp binding, frame edge, out-of-gamut.
"""
import os, sys
import numpy as np, torch
from PIL import Image
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0, HERE)
from lapfuse import lap_pyr, lap_recon, boxf, _K
from energy_restore import restore
Image.MAX_IMAGE_PIXELS = None
dev = "cuda"
ld = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(dev)

T = "HCM0421"
ENS = f"/mnt/d/avv/r29/tower_ens/{T}/png_ens"
R22 = f"/mnt/d/avv/r22/tower_ens/{T}/png_ens"
MEM = [f"/mnt/d/avv/r2r9/models/{T}_ut7/test_png", f"/mnt/d/avv/r2r9/models/{T}_ut13/test_png",
       f"/mnt/d/avv/r2r9/models/{T}_ut42/test_png", f"/mnt/d/avv/r2r9/models/{T}_ut77/test_png",
       f"/mnt/d/avv/r22_seed101/{T}/test_png", f"/mnt/d/avv/r25_mip3d/{T}/test_png",
       f"/mnt/d/avv/r28_members/{T}/test_png"]
stems = sorted(os.path.splitext(f)[0] for f in os.listdir(ENS) if f.endswith(".png"))
print(f"{T}: {len(stems)} stems")

s = stems[0]
e = ld(os.path.join(ENS, s + ".png"))
ms = [ld(os.path.join(d, s + ".png")) for d in MEM]
r22 = ld(os.path.join(R22, s + ".png"))

# ---------------- (a) exactness of analysis/synthesis
K = _K.to(dev)
laps, res, sizes = lap_pyr(e, 5, K)
rec = lap_recon(laps, res, sizes, K)
print(f"\n(a) lap_recon round-trip float32 nlev=5: max|err| = {(rec-e).abs().max().item():.3e}  "
      f"rms = {(rec-e).pow(2).mean().sqrt().item():.3e}   (1 LSB = {1/255:.4e})")
e64 = e.double(); K64 = _K.double().to(dev)
l64, r64, s64 = lap_pyr(e64, 5, K64)
print(f"    float64 nlev=5: max|err| = {(lap_recon(l64,r64,s64,K64)-e64).abs().max().item():.3e}")

# ---------------- (b) does nlev matter?
o5 = restore(e, ms, 1.0, 8, 3, 5, 4.0)
o1 = restore(e, ms, 1.0, 8, 3, 1, 4.0)
print(f"\n(b) restore nlev=5 vs nlev=1: max|diff| = {(o5-o1).abs().max().item():.3e} "
      f"({(o5-o1).abs().max().item()*255:.2e} LSB)  -> nlev is a NO-OP, costs {5}x the pyramid work")
def restore64(ens, mems, lam, k, win, nlev, clamp):
    KK = _K.double().to(ens.device)
    lp, rs, sz = lap_pyr(ens, nlev, KK); L0 = lp[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = sum(boxf(((lap_pyr(m, nlev, KK)[0][0] - L0) ** 2).sum(1, keepdim=True), win) for m in mems)
    V = V / len(mems) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    return lap_recon([L0 * (1.0 + lam * (r - 1.0))] + lp[1:], rs, sz, KK)
o5_64 = restore64(e.double(), [m.double() for m in ms], 1.0, 8, 3, 5, 4.0)
print(f"    float32 vs float64 restore: max|diff| = {(o5.double()-o5_64).abs().max().item():.3e} "
      f"({(o5.double()-o5_64).abs().max().item()*255:.3e} LSB)")

# ---------------- (c) what did production actually average?
print("\n(c) DECOMPOSING THE SHIPPED ENSEMBLE (uint8 png_ens, tol = 1 LSB)")
sum5 = sum(ms[:5])
for n in (5, 6, 7, 8):
    X = n * r22 - sum5                     # sum of the (n-5) unknown members
    print(f"    r22 png_ens as {n}-way mean -> residual sum of {n-5} unknown member(s): "
          f"min {X.min().item():+.3f} max {X.max().item():+.3f}  "
          f"(valid iff in [0,{max(n-5,0)}])")
for lab, w in (("4:1:1 (as coded)", (4, 1, 1)), ("6:1:1 (true uniform 8)", (6, 1, 1)),
               ("7:1:1 (true uniform 9)", (7, 1, 1))):
    tot = sum(w)
    cand = (w[0] * r22 + w[1] * ms[5] + w[2] * ms[6]) / tot
    d = (cand - e).abs()
    print(f"    r29 png_ens vs weights {lab:24s}: max {d.max().item()*255:7.3f} LSB, "
          f"mean {d.mean().item()*255:7.4f} LSB")

# ---------------- (d) r-map statistics on real shipped data
print("\n(d) r-MAP STATISTICS ON REAL SHIPPED DATA (lam=1, k=8, win=3, clamp=4)")
NS = 8
for TT in ("HCM0421", "HCM0539", "HCM0674"):
    E2 = f"/mnt/d/avv/r29/tower_ens/{TT}/png_ens"
    M2 = [d.replace(T, TT) for d in MEM]
    st = sorted(os.path.splitext(f)[0] for f in os.listdir(E2) if f.endswith(".png"))[::len(os.listdir(E2))//NS or 1][:NS]
    acc = dict(clamp=0., rmean=0., rmax=0., edge=0., inter=0., clip=0., disag=0., n=0,
               boost_rms=0., q999=0.)
    for ss in st:
        ee = ld(os.path.join(E2, ss + ".png"))
        mm = [ld(os.path.join(d, ss + ".png")) for d in M2]
        L0 = lap_pyr(ee, 1, K)[0][0]
        Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
        V = sum(boxf(((lap_pyr(m, 1, K)[0][0] - L0) ** 2).sum(1, keepdim=True), 3) for m in mm)
        V = V / len(mm) * (8 / 7.)
        rr = torch.sqrt(1.0 + V / (Eb + 1e-10))
        rc = rr.clamp(max=4.0)
        out = ee + 1.0 * (rc - 1.0) * L0
        acc["clamp"] += float((rr > 4.0).float().mean())
        acc["rmean"] += float(rc.mean()); acc["rmax"] += float(rr.max())
        acc["q999"] += float(torch.quantile(rr.flatten().float()[::7], 0.999))
        B = 8
        m_edge = torch.zeros_like(rc, dtype=torch.bool); m_edge[..., :B, :] = 1
        m_edge[..., -B:, :] = 1; m_edge[..., :, :B] = 1; m_edge[..., :, -B:] = 1
        acc["edge"] += float(rc[m_edge].mean()); acc["inter"] += float(rc[~m_edge].mean())
        acc["clip"] += float(((out < 0) | (out > 1)).float().mean())
        acc["boost_rms"] += float(((rc - 1.0) * L0).pow(2).mean().sqrt())
        acc["disag"] += float(torch.stack([(m - ee).abs().mean() for m in mm]).mean()) * 255
        acc["n"] += 1
    n = acc["n"]
    print(f"  {TT}: mean r {acc['rmean']/n:.4f} | r>4 (clamped) {100*acc['clamp']/n:.4f}% | "
          f"max r {acc['rmax']/n:.1f} | p99.9 r {acc['q999']/n:.2f} | "
          f"edge8 r {acc['edge']/n:.4f} vs interior {acc['inter']/n:.4f} "
          f"({100*(acc['edge']/n/(acc['inter']/n)-1):+.2f}%) | out-of-gamut {100*acc['clip']/n:.4f}% | "
          f"boost rms {255*acc['boost_rms']/n:.3f} LSB | member disagreement {acc['disag']/n:.2f}/255")

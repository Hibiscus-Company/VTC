"""ADV lens-1 CONFOUND probe. Builds candidate dirs + free (no-GPU) diversity/HF stats.
Q: is the +0.3615 'ensemble edge' a variance-reduction property of the SHIPPED pool
   (6 SEEDS of ONE recipe, EXPERIMENTS.md:3217) or an artefact of a CROSS-RECIPE eval pool
   (6 different recipes) plus plain smoothing?"""
import os, itertools, numpy as np
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = None
B = "/mnt/d/avv/bonsai_eval"; GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
OUT = "/mnt/d/avv/ADVC"; os.makedirs(OUT, exist_ok=True)
ARMS6 = ["K1_noUT_aa", "K4_pC_seed1k", "K4_pC_seed7", "eps10", "eps20", "ppisp_pc"]
PAIR = ["K4_pC_seed7", "K4_pC_seed1k"]            # same recipe, different seed = SHIPPED axis
stems = sorted(f[:-4] for f in os.listdir(f"{B}/{ARMS6[0]}/eval_png") if f.endswith(".png"))
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
dirs = {"mean6": f"{OUT}/mean6", "meanpair_seed": f"{OUT}/meanpair_seed",
        "K1_blur05": f"{OUT}/K1_blur05", "K1_blur08": f"{OUT}/K1_blur08"}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

def grad(a):  # HF energy proxy, LSB units
    return float(np.sqrt(((np.diff(a, axis=0) ** 2).mean() + (np.diff(a, axis=1) ** 2).mean()) / 2))

dis = {p: [] for p in itertools.combinations(ARMS6, 2)}
hf = {k: [] for k in ARMS6 + ["mean6", "meanpair_seed", "K1_blur05", "K1_blur08", "GT"]}
for s in stems:
    A = {a: np.asarray(Image.open(f"{B}/{a}/eval_png/{s}.png").convert("RGB"), dtype=np.float64) for a in ARMS6}
    for p in dis:
        dis[p].append(float(np.sqrt(((A[p[0]] - A[p[1]]) ** 2).mean())))
    m6 = np.mean([A[a] for a in ARMS6], 0)
    mp = np.mean([A[a] for a in PAIR], 0)
    Image.fromarray(np.clip(m6 + 0.5, 0, 255).astype(np.uint8)).save(f"{dirs['mean6']}/{s}.png")
    Image.fromarray(np.clip(mp + 0.5, 0, 255).astype(np.uint8)).save(f"{dirs['meanpair_seed']}/{s}.png")
    k1 = Image.open(f"{B}/K1_noUT_aa/eval_png/{s}.png").convert("RGB")
    for tag, sg in (("K1_blur05", 0.5), ("K1_blur08", 0.8)):
        b = k1.filter(ImageFilter.GaussianBlur(radius=sg))
        b.save(f"{dirs[tag]}/{s}.png")
        hf[tag].append(grad(np.asarray(b, dtype=np.float64).mean(2)))
    for a in ARMS6:
        hf[a].append(grad(A[a].mean(2)))
    hf["mean6"].append(grad(m6.mean(2)))
    hf["meanpair_seed"].append(grad(mp.mean(2)))
    hf["GT"].append(grad(np.asarray(Image.open(f"{GT}/{gt_by[s]}").convert("RGB"), dtype=np.float64).mean(2)))
print("=== pairwise RMS disagreement (LSB), n=28 frames ===")
for p, v in sorted(dis.items(), key=lambda kv: np.mean(kv[1])):
    tag = "SAME-RECIPE/SEED" if set(p) == set(PAIR) else "cross-recipe"
    print(f"  {p[0]:14s} vs {p[1]:14s}  {np.mean(v):6.3f}   {tag}")
same = np.mean(dis[tuple(PAIR)] if tuple(PAIR) in dis else dis[(PAIR[1], PAIR[0])])
cross = np.mean([np.mean(v) for p, v in dis.items() if set(p) != set(PAIR)])
print(f"  SAME-RECIPE seed pair {same:.3f}  vs  cross-recipe mean {cross:.3f}   ratio {cross/same:.2f}x"
      f"   var ratio {(cross/same)**2:.2f}x")
print("=== HF energy (mean |grad| RMS, LSB) ===")
for k, v in hf.items():
    print(f"  {k:16s} {np.mean(v):7.3f}")
np.save(f"{OUT}/dis.npy", np.array([[np.mean(dis[(a, b)]) if (a, b) in dis else (np.mean(dis[(b, a)]) if (b, a) in dis else 0.0) for b in ARMS6] for a in ARMS6]))
print("dirs:", " ".join(dirs.values()))

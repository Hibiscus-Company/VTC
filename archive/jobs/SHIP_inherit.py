"""SHIP-PATH: does an ensemble inherit a member-quality shift?
Two designs, both on the 28 bonsai eval holes, today's scripts/eval_score.py.

T1 WHOLE-POOL SHIFT (k=3): mean(3 best arms) vs mean(3 worst arms).
    single-mean delta is known; inheritance = ens_delta / single_delta.
    This is the "all 6 members retrained at the new setting" scenario, at k=3.

T2 MARGINAL 1-of-6 SWAP with the MATCHED PAIR (same recipe, same seed 42, one variable):
    E_ctrl = mean(sr001seed42 + 5 arms)   vs   E_new = mean(sr0.03 + same 5 arms)
    naive expectation for ANY member change is delta_single/6 = 0.0295.
    Coming in ABOVE that means the new member also decorrelates; BELOW means it duplicates.

Also prints pairwise disagreement RMS (LSB) per group -- the diversity confound for T1.
"""
import os, sys, subprocess, tempfile, itertools, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ES = "/mnt/d/avv/evalsplit/bonsai"
B = "/mnt/d/avv/bonsai_eval"
REPO = "/mnt/c/Users/BKAI/an_plaza2/FastGS"

DIRS = {
    "K1_noUT_aa":   f"{B}/K1_noUT_aa/eval_png",
    "K4_pC_seed1k": f"{B}/K4_pC_seed1k/eval_png",
    "K4_pC_seed7":  f"{B}/K4_pC_seed7/eval_png",
    "eps10":        f"{B}/eps10/eval_png",
    "eps20":        f"{B}/eps20/eval_png",
    "ppisp_pc":     f"{B}/ppisp_pc/eval_png",
    "sr001seed42":  "/mnt/d/avv/r36_shape/sr001seed42/eval_png",
    "sr0.03":       "/mnt/d/avv/r35_scalereg/sr0.03/eval_png",
    "aniso001":     "/mnt/d/avv/r36_shape/aniso001/eval_png",
}
SINGLE = {"K1_noUT_aa": 71.9029, "K4_pC_seed1k": 71.3067, "K4_pC_seed7": 71.2190,
          "eps10": 71.5985, "eps20": 71.5260, "ppisp_pc": 71.2285,
          "sr001seed42": 71.6951, "sr0.03": 71.8718, "aniso001": 71.6575}

STEMS = sorted(f[:-4] for f in os.listdir(DIRS["K1_noUT_aa"]) if f.endswith(".png"))
assert len(STEMS) == 28, len(STEMS)

_cache = {}
def arr(name, stem):
    k = (name, stem)
    if k not in _cache:
        _cache[k] = np.asarray(Image.open(f"{DIRS[name]}/{stem}.png").convert("RGB"), dtype=np.float32)
    return _cache[k]

def score_mean(names, tag):
    """float mean over members, round once (matches ensemble_renders.py / bar.py)."""
    tmp = tempfile.mkdtemp(prefix=f"SH_{tag}_", dir="/mnt/d/avv")
    for s in STEMS:
        a = np.mean([arr(n, s) for n in names], axis=0)
        Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8)).save(f"{tmp}/{s}.png")
    r = subprocess.run([sys.executable, "scripts/eval_score.py", "--render_dir", tmp,
                        "--gt_dir", f"{ES}/eval_gt", "--tag", tag],
                       cwd=REPO, capture_output=True, text=True)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in (r.stdout + r.stderr).splitlines() if "SCORE" in l]
    if not line:
        print("FAIL", tag, (r.stdout + r.stderr)[-400:]); return None
    print("   ", line[-1].strip())
    return float(line[-1].split("SCORE")[1].split()[0])

def disagree(names):
    """mean over frames of RMS pairwise pixel difference, in LSB."""
    v = []
    for s in STEMS[:8]:
        for a, b in itertools.combinations(names, 2):
            v.append(np.sqrt(((arr(a, s) - arr(b, s)) ** 2).mean()))
    return float(np.mean(v))

BEST3  = ["K1_noUT_aa", "eps10", "eps20"]
WORST3 = ["K4_pC_seed1k", "K4_pC_seed7", "ppisp_pc"]
FIVE   = ["K1_noUT_aa", "eps10", "eps20", "K4_pC_seed1k", "ppisp_pc"]  # 5 held fixed for T2

print("=== T1  WHOLE-POOL QUALITY SHIFT, k=3 ===")
sb, sw = np.mean([SINGLE[n] for n in BEST3]), np.mean([SINGLE[n] for n in WORST3])
print(f"  BEST3  singles mean {sb:.4f}   disagreement {disagree(BEST3):.3f} LSB")
eb = score_mean(BEST3, "best3")
print(f"  WORST3 singles mean {sw:.4f}   disagreement {disagree(WORST3):.3f} LSB")
ew = score_mean(WORST3, "worst3")
if eb and ew:
    print(f"  >>> single delta {sb-sw:+.4f}   ENSEMBLE delta {eb-ew:+.4f}   "
          f"INHERITANCE {(eb-ew)/(sb-sw):.3f}")

print("\n=== T2  MARGINAL 1-of-6 SWAP, matched pair (seed 42, one variable) ===")
print(f"  member single delta sr0.03 - sr001seed42 = {SINGLE['sr0.03']-SINGLE['sr001seed42']:+.4f}"
      f"   (naive 1/6 -> {(SINGLE['sr0.03']-SINGLE['sr001seed42'])/6:+.4f})")
print(f"  disagreement ctrl-vs-5 {disagree(['sr001seed42']+FIVE):.3f} LSB   "
      f"new-vs-5 {disagree(['sr0.03']+FIVE):.3f} LSB")
ec = score_mean(["sr001seed42"] + FIVE, "ens_ctrl")
en = score_mean(["sr0.03"] + FIVE, "ens_new")
if ec and en:
    d = SINGLE["sr0.03"] - SINGLE["sr001seed42"]
    print(f"  >>> ENSEMBLE delta {en-ec:+.4f}   vs naive k-scaled {d/6:+.4f}   "
          f"ratio-to-naive {(en-ec)/(d/6):.2f}")

print("\n=== reference: 6-arm baseline mean (the quoted BAR) ===")
score_mean(["K1_noUT_aa", "K4_pC_seed1k", "K4_pC_seed7", "eps10", "eps20", "ppisp_pc"], "bar6")

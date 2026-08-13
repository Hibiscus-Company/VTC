"""Phase 8: does the cross-family DECORRELATION that pays off on HCM0181 exist on the
five GRADED private_set2 towers, using models that ALREADY EXIST (zero GPU training)?

We cannot score private_set2 (no test GT), so we measure the one quantity that predicted
contribution GT-free on HCM0181 (R^2 = 0.949 over the competitive members): the mean pixel
distance from a candidate member to the existing base ensemble members, expressed as a
RATIO to the within-family (seed-twin) distance so it is comparable across scenes.
"""
import os, sys, json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"

TOWERS = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]
BASE_T = {t: [f"/mnt/d/avv/r2r9/models/{t}_{s}/test_png" for s in ("ut7", "ut42", "ut13", "ut77")]
          for t in TOWERS}
CAND_T = {t: {**{f"s2gates_{m}": f"/mnt/d/avv/output_s2gates/{t}_{m}/test_png" for m in ("champA", "memB", "memC")},
              "seed101": f"/mnt/d/avv/r22_seed101/{t}/test_png",
              "mip3d": f"/mnt/d/avv/r25_mip3d/{t}/test_png"} for t in TOWERS}


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def dist_stats(dirs_base, dirs_cand):
    """Returns within-base mean RMS distance and cand->base mean RMS distance."""
    stems = None
    for d in list(dirs_base) + list(dirs_cand.values()):
        s = {os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(".png")}
        stems = s if stems is None else (stems & s)
    stems = sorted(stems)
    nb = len(dirs_base)
    wsum = np.zeros(1); wn = 0
    csum = {k: 0.0 for k in dirs_cand}; cn = 0
    for st in stems:
        B = [load(os.path.join(d, st + ".png")) for d in dirs_base]
        for i in range(nb):
            for j in range(i + 1, nb):
                wsum += np.sqrt(((B[i] - B[j]) ** 2).mean()); wn += 1
        for k, d in dirs_cand.items():
            c = load(os.path.join(d, st + ".png"))
            csum[k] += float(np.mean([np.sqrt(((c - b) ** 2).mean()) for b in B]))
        cn += 1
    return float(wsum[0] / wn), {k: v / cn for k, v in csum.items()}, len(stems)


def main():
    # --- HCM0181 calibration: same statistic on the scene where we HAVE scores
    R = np.load(os.path.join(TMP, "renders_u8.npy"), mmap_mode="r")
    N = open(os.path.join(TMP, "names.txt")).read().split()[:21]
    ui = [N.index(x) for x in ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]]
    D = np.load(os.path.join(HERE, "p2_gram.npz"), allow_pickle=True)["ddiff"].mean(0)
    within = np.mean([D[i, j] for i in ui for j in ui if i < j])
    print("=== HCM0181 (scored) reference: within-UT4 mean RMS distance = %.5f" % within)
    print(f"{'member':18s} {'D_to_base':>10s} {'ratio':>7s}   measured contribution")
    p3 = json.load(open(os.path.join(HERE, "p3.json")))
    b0 = p3["B_addone"]["base_UT4"][0]
    for nm in ["m31b_taillpips", "gsplatB2", "gsplatB8pure", "e17visnorm", "e15ceil95", "e16app"]:
        i = N.index(nm); db = float(np.mean([D[i, j] for j in ui]))
        print(f"{nm:18s} {db:10.5f} {db/within:7.3f}   {p3['B_addone']['add5_'+nm][0]-b0:+.4f}")

    # --- graded towers
    out = {}
    print("\n=== private_set2 graded towers (real test poses, no GT available)")
    print(f"{'scene':9s} {'within-UT':>10s} {'member':16s} {'D_to_UT':>9s} {'ratio':>7s}")
    for t in TOWERS:
        cd = {k: v for k, v in CAND_T[t].items() if os.path.isdir(v)}
        w, c, ns = dist_stats(BASE_T[t], cd)
        out[t] = {"within": w, "cand": c, "n": ns}
        for k in sorted(c, key=lambda k: -c[k]):
            print(f"{t:9s} {w:10.5f} {k:16s} {c[k]:9.5f} {c[k]/w:7.3f}")
    json.dump(out, open(os.path.join(HERE, "p8.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

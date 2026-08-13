"""STEP 1: score every HCM0181 member variant singly, on the production harness."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_lib as A

VARIANTS = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
            "gsplatB1", "gsplatB2", "gsplatB3", "gsplatB4warm", "gsplatB5affine",
            "gsplatB6bilagrid", "gsplatB7ppisp2", "gsplatB8pure",
            "e15ceil95", "e16app", "e17visnorm",
            "m31b_nolpips", "m31b_taillpips", "sh0", "sh1", "sh2", "sh3"]


def main():
    dev = A.init()
    stems, gt_by = A.stems_for(VARIANTS[:1])  # union stems = those with GT
    # keep only stems present in ALL variants for a fair comparison
    keep = [s for s in stems if all(os.path.exists(os.path.join(A.mdir(v), s + ".png"))
                                    for v in VARIANTS)]
    print(f"stems common to all {len(VARIANTS)} variants: {len(keep)} (GT-only stems: {len(stems)})")
    sc = A.Scorer(dev)
    for i, s in enumerate(keep):
        g = A.gt_tensor(gt_by[s], dev)
        for v in VARIANTS:
            sc.add(v, A.load(A.mdir(v), s), g)
        del g
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(keep)}", flush=True)
    rows = sorted(sc.table(), key=lambda r: -r[1])
    print(f"\n{'variant':<18}{'SCORE':>9}{'PSNR':>9}{'SSIM':>8}{'LPIPS':>9}")
    for k, s, P, S, L, n in rows:
        print(f"{k:<18}{s:9.4f}{P:9.4f}{S:8.4f}{L:9.4f}")


if __name__ == "__main__":
    main()

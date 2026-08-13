"""Aggregate the harness sweeps + leave-one-SCENE-out CV of the field-scale constant."""
import json, sys
import numpy as np

RES = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/res"
PUB = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
SC = [1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0]


def table(path, keys=None, ref="none"):
    d = json.load(open(path))
    scenes = [s for s in PUB if s in d]
    keys = keys or list(d[scenes[0]].keys())
    ns = np.array([d[s][keys[0]]["n"] for s in scenes], float)
    print(f"\n### {path.split('/')[-1]}   scenes={scenes}  n={int(ns.sum())}")
    hdr = f"{'variant':16s}" + "".join(f"{s:>11s}" for s in scenes) + f"{'MEAN d':>11s}"
    print(hdr)
    for k in keys:
        ds = np.array([d[s][k]["score"] - d[s][ref]["score"] for s in scenes])
        w = (ds * ns).sum() / ns.sum()
        print(f"{k:16s}" + "".join(f"{x:+11.4f}" for x in ds) + f"{w:+11.4f}")
    return d, scenes, ns


def loso(d, scenes, ns):
    """leave-one-SCENE-out selection of the global scale constant"""
    key = lambda a: f"med_s{a}"
    delta = {s: {a: d[s][key(a)]["score"] - d[s]["med_s1.0"]["score"] for a in SC} for s in scenes}
    print("\n### leave-one-SCENE-out CV of the scale constant (delta vs the shipped scale 1.0)")
    tot, wt = 0.0, 0.0
    for i, s in enumerate(scenes):
        rest = [t for t in scenes if t != s]
        pick = max(SC, key=lambda a: np.mean([delta[t][a] for t in rest]))
        got = delta[s][pick]
        print(f"  hold out {s:9s}: 4-scene pick = {pick:4.2f}   delivers on {s}: {got:+.4f}")
        tot += got * ns[i]; wt += ns[i]
    print(f"  --> honest image-weighted CV gain of the procedure: {tot/wt:+.4f}")
    best = max(SC, key=lambda a: (np.array([delta[s][a] for s in scenes]) * ns).sum() / ns.sum())
    allg = (np.array([delta[s][best] for s in scenes]) * ns).sum() / ns.sum()
    print(f"  --> in-sample best constant {best} = {allg:+.4f}")
    print("\n  per-scene curves (delta vs scale 1.0):")
    print(f"  {'scene':10s}" + "".join(f"{a:>9.2f}" for a in SC))
    for s in scenes:
        print(f"  {s:10s}" + "".join(f"{delta[s][a]:+9.4f}" for a in SC))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "scale2"
    if what == "scale2":
        d, scenes, ns = table(f"{RES}/scale2.json")
        loso(d, scenes, ns)
    else:
        table(f"{RES}/{what}.json")

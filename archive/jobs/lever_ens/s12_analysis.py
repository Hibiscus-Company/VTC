"""Final analysis: greedy/rank curves, per-step CV of the greedy selection rule,
CV + bootstrap of the marginal 'add-one' decisions, weight-sweep CV."""
import os, sys, json, numpy as np

OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens"


def summ(P, S, L):
    l, s, p = np.mean(L), np.mean(S), np.mean(P)
    return 100 * (0.4 * (1 - l) + 0.3 * s + 0.3 * min(p / 50, 1.0))


def sub(per, imgs):
    pos = {g: k for k, g in enumerate(per["imgs"])}
    k = [pos[g] for g in imgs]
    return summ(np.array(per["psnr"])[k], np.array(per["ssim"])[k], np.array(per["lpips"])[k])


rng = np.random.RandomState(0); perm = rng.permutation(60)
FA, FB = sorted(perm[:30].tolist()), sorted(perm[30:].tolist())
rng2 = np.random.RandomState(7); p2 = rng2.permutation(60)
FA2, FB2 = sorted(p2[:30].tolist()), sorted(p2[30:].tolist())
ALL = list(range(60))
singles = json.load(open(os.path.join(OUT, "singles.json")))
FAMOF = {"e15ceil95": "FastGS", "e16app": "FastGS", "e17visnorm": "FastGS",
         "gsplatB1": "MCMC", "gsplatB2": "MCMC", "gsplatB3": "MCMC", "gsplatB4warm": "MCMC",
         "gsplatB5affine": "MCMC", "gsplatB6bilagrid": "MCMC", "gsplatB7ppisp2": "MCMC",
         "gsplatB8pure": "MCMC", "gsplatB9ut": "UT", "gsplatB10ut8M": "UT",
         "gsplatB11ut60k": "UT", "gsplatB12ut8Ms7": "UT", "m31b_nolpips": "UTmet",
         "m31b_taillpips": "UTmet", "sh0": "UTsh", "sh1": "UTsh", "sh2": "UTsh"}

# ---------------- TABLE 1 ----------------
log = json.load(open(os.path.join(OUT, "greedy.json")))
print("=== TABLE 1: ensemble score vs k  (60 real test views, project scorer) ===")
print(f"{'k':>2s} | {'RANK-ORDER path':45s} | {'GREEDY path (in-sample selection)':50s}")
print(f"{'':>2s} | {'added':17s}{'fam':6s}{'score':>9s}{'marg':>8s} | {'added':17s}{'fam':6s}{'score':>9s}{'marg':>8s}{'single':>9s}")
pr = pg = None
for i in range(len(log["rank"])):
    r = log["rank"][i]
    mr = "" if pr is None else f"{r['score'] - pr:+.4f}"
    line = f"{r['k']:2d} | {r['added']:17s}{FAMOF[r['added']]:6s}{r['score']:9.4f}{mr:>8s} |"
    pr = r["score"]
    if i < len(log["greedy"]):
        g = log["greedy"][i]
        mg = "" if pg is None else f"{g['score']-pg:+8.4f}"
        line += f" {g['added']:17s}{FAMOF[g['added']]:6s}{g['score']:9.4f}{mg:>8s}{g['single']:9.4f}"
        pg = g["score"]
    print(line)

# ---------------- TABLE 2: greedy per-step CV ----------------
cpath = os.path.join(OUT, "cache_per.npy")
if os.path.exists(cpath):
    cache = np.load(cpath, allow_pickle=True).item()
    POOL = [n for n in singles if n not in ("sh3", "k4")]
    print("\n=== TABLE 2: is the GREEDY selection real, or fitted to these 60 views? ===")
    print("   at each step: pick by fold-A(30 views), report the marginal on held-out fold-B(30 views)")
    print(f"{'k':>2s} {'full-60 pick':18s} {'fold-A pick':18s} {'same':>5s} {'heldout marg(A-pick)':>21s} {'in-sample marg':>15s}")
    cur = []
    agree = tot = 0; hg = []
    for step in log["greedy"]:
        prev = tuple(sorted(cur))
        if cur and prev not in cache:
            cur.append(step["added"]); continue
        cands = {c: cache[tuple(sorted(cur + [c]))]["per"] for c in POOL
                 if c not in cur and tuple(sorted(cur + [c])) in cache}
        if len(cands) < 2:
            cur.append(step["added"]); continue
        bp = cache[prev]["per"] if cur else None
        pA = max(cands, key=lambda c: sub(cands[c], FA))
        dB = sub(cands[pA], FB) - (sub(bp, FB) if bp else 0)
        ins = step["score"] - (log["greedy"][len(cur) - 1]["score"] if cur else 0)
        if cur:
            print(f"{step['k']:2d} {step['added']:18s} {pA:18s} {str(pA==step['added']):>5s} "
                  f"{dB:+21.4f} {ins:+15.4f}")
            agree += (pA == step["added"]); tot += 1; hg.append(dB)
        cur.append(step["added"])
    print(f"   fold-A and full-60 pick the SAME member in {agree}/{tot} steps; "
          f"cumulative held-out marginal over steps 2..{tot+1} = {np.sum(hg):+.4f}")

# ---------------- TABLE 3: add-one-to-k4 with CV + bootstrap ----------------
mpath = os.path.join(OUT, "margper.json")
if os.path.exists(mpath):
    mp = json.load(open(mpath))
    b = mp["__base__"]["per"]
    corr = {r["name"]: r for r in json.load(open(os.path.join(OUT, "corr.json")))}
    print("\n=== TABLE 3: marginal value of ADDING one member to the calibrated k4 base ===")
    print("   (uniform 1/5 weights; base k4 = %.4f)" % summ(b["psnr"], b["ssim"], b["lpips"]))
    print(f"{'add':18s}{'fam':7s}{'single':>8s}{'gap':>7s}{'corr':>6s}{'delta60':>9s}"
          f"{'boot SE':>8s}{'foldA':>8s}{'foldB':>8s}{'split2 A/B':>16s}")
    rows = []
    for x, d in mp.items():
        if x == "__base__":
            continue
        p = d["per"]
        d60 = summ(p["psnr"], p["ssim"], p["lpips"]) - summ(b["psnr"], b["ssim"], b["lpips"])
        r = np.random.RandomState(3)
        bs = []
        for _ in range(2000):
            idx = r.randint(0, 60, 60)
            bs.append(sub(p, [ALL[i] for i in idx]) - sub(b, [ALL[i] for i in idx]))
        se = float(np.std(bs))
        dA, dB = sub(p, FA) - sub(b, FA), sub(p, FB) - sub(b, FB)
        dA2, dB2 = sub(p, FA2) - sub(b, FA2), sub(p, FB2) - sub(b, FB2)
        rows.append((d60, x, se, dA, dB, dA2, dB2))
    for d60, x, se, dA, dB, dA2, dB2 in sorted(rows, reverse=True):
        print(f"{x:18s}{FAMOF[x]:7s}{singles[x]['score']:8.4f}"
              f"{singles[x]['score']-75.9644:+7.3f}{corr[x]['corr_k4']:6.3f}{d60:+9.4f}"
              f"{se:8.4f}{dA:+8.4f}{dB:+8.4f}{dA2:+8.4f}{dB2:+8.4f}")
    sg = [r for r in rows if r[0] > 0]
    print(f"   sign agreement across the 4 half-splits: "
          f"{sum(1 for r in rows if np.sign(r[3])==np.sign(r[4])==np.sign(r[5])==np.sign(r[6])==np.sign(r[0]))}/{len(rows)}")

# ---------------- TABLE 4: weight sweep ----------------
wpath = os.path.join(OUT, "wsweep.json")
if os.path.exists(wpath):
    ws = json.load(open(wpath))
    bp = ws["base_per"]; b0 = summ(bp["psnr"], bp["ssim"], bp["lpips"])
    print("\n=== TABLE 4: weight sweep for ONE added member on the k4 base (base %.4f) ===" % b0)
    print("   w = weight on the added member; the 4 base members share (1-w)")
    byadd = {}
    for r in ws["rows"]:
        byadd.setdefault(r["add"], []).append(r)
    print(f"{'add':18s}" + "".join(f"{'w='+format(w,'.2f'):>10s}" for w in
                                   [x["w"] for x in byadd[list(byadd)[0]]]) +
          f"{'argmax w':>10s}{'foldA->B':>10s}")
    for x, rs in byadd.items():
        deltas = [r["score"] - b0 for r in rs]
        best = int(np.argmax(deltas))
        pickA = int(np.argmax([sub(r["per"], FA) - sub(bp, FA) for r in rs]))
        dB = sub(rs[pickA]["per"], FB) - sub(bp, FB)
        print(f"{x:18s}" + "".join(f"{d:+10.4f}" for d in deltas) +
              f"{rs[best]['w']:10.2f}{dB:+10.4f}")

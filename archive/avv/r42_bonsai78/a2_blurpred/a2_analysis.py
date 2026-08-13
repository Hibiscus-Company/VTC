#!/usr/bin/env python
import os, sys, csv, json
import numpy as np
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from blur_predictor import BlurPredictor, STRIDE

OUT = "/mnt/d/avv/r42_bonsai78/a2_blurpred"
SPLIT = json.load(open("/mnt/d/avv/evalsplit/bonsai/split.json"))
EVAL = sorted(int(x) for x in SPLIT["eval_frames"])


def load_csv(p):
    d = {}
    for r in csv.DictReader(open(p)):
        d[int(r["frame"])] = {k: float(v) for k, v in r.items() if k != "name"}
    return d


T = load_csv(f"{OUT}/train248_sharp.csv")          # all 248 train photos (GT sharpness)
R = load_csv(f"{OUT}/render_sr01_sharp.csv")       # 28 renders
ALL = sorted(T)
SUB = [f for f in ALL if f not in set(EVAL)]       # 220 train_sub
assert len(SUB) == 220 and len(EVAL) == 28

# real private-test adjacency: 5 adjacent pairs -> 10 frames with one neighbour missing
TESTF = [80, 90, 280, 320, 370, 380, 530, 550, 740, 880, 930, 980, 1010, 1050, 1590,
         1750, 1870, 1880, 1950, 1970, 2030, 2100, 2390, 2590, 2600, 2620, 2630, 2740]
adj = sorted(f for f in TESTF if (f - 10) in TESTF or (f + 10) in TESTF)
print("real test frames with an adjacent test frame:", len(adj), adj)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


# ============================================================= STEP 2
print("\n" + "=" * 96)
print("STEP 2  -- predict eval-hole sharpness from train_sub (220) only")
print("=" * 96)

# For the eval holes: all 28 are isolated (min gap between eval frames):
gaps = np.diff(EVAL)
print(f"eval-hole spacing: min {gaps.min()} -> every eval hole has BOTH +-10 neighbours present.")
print("=> the one-neighbour case is produced by additionally hiding one +-10 neighbour.")

STATS = ["log_lapvar", "log_hf025", "reblur"]
MODES = [("nn2", None), ("nw", "grid"), ("ll", "grid"), ("lq", "grid")]
# explicit fixed-window local poly requested in the task
FIXED = [("ll", 30.0), ("ll", 50.0), ("lq", 30.0), ("lq", 50.0)]

results = {}
for st in STATS:
    obs = {f: T[f][st] for f in SUB}
    y = np.array([T[f][st] for f in EVAL])
    print(f"\n--- target = {st}   (eval-hole sd = {y.std(ddof=1):.4f}, "
          f"train_sub sd = {np.std(list(obs.values()), ddof=1):.4f})")
    rows = []
    for mode, kind in MODES:
        if kind == "grid":
            bp = BlurPredictor.fit(obs, mode=mode)
            h = bp.h
        else:
            bp = BlurPredictor(list(obs), list(obs.values()), h=1.0, mode=mode)
            h = np.nan
            bp.loo_r2 = r2(bp.fy, bp.loo())
        # both-neighbour prediction
        pb = bp.predict_many(EVAL)
        # one-neighbour: hide the +10 neighbour (worst realistic case)
        po = bp.predict_many(EVAL, {f: (f + STRIDE,) for f in EVAL})
        # one-neighbour, other side
        pm = bp.predict_many(EVAL, {f: (f - STRIDE,) for f in EVAL})
        p1 = np.concatenate([po, pm]); y1 = np.concatenate([y, y])
        rows.append((mode, h, bp.loo_r2, r2(y, pb), rmse(y, pb), r2(y1, p1), rmse(y1, p1)))
    for mode, h in FIXED:
        bp = BlurPredictor(list(obs), list(obs.values()), h=h, mode=mode)
        bp.loo_r2 = r2(bp.fy, bp.loo())
        pb = bp.predict_many(EVAL)
        po = bp.predict_many(EVAL, {f: (f + STRIDE,) for f in EVAL})
        pm = bp.predict_many(EVAL, {f: (f - STRIDE,) for f in EVAL})
        p1 = np.concatenate([po, pm]); y1 = np.concatenate([y, y])
        rows.append((mode + "@fix", h, bp.loo_r2, r2(y, pb), rmse(y, pb), r2(y1, p1), rmse(y1, p1)))
    print(f"  {'estimator':10s} {'h':>6s} {'LOO_R2(sub)':>11s} | {'R2 both':>8s} {'RMSE':>7s} "
          f"| {'R2 one':>8s} {'RMSE':>7s}")
    for m, h, l, rb_, eb, ro, eo in rows:
        print(f"  {m:10s} {h:6.1f} {l:11.3f} | {rb_:8.3f} {eb:7.4f} | {ro:8.3f} {eo:7.4f}")
    # keep the best grid-fitted local-linear predictor for later steps
    bp = BlurPredictor.fit(obs, mode="ll")
    results[st] = dict(bp=bp, pred_both=bp.predict_many(EVAL),
                       pred_one=bp.predict_many(EVAL, {f: (f + STRIDE,) for f in EVAL}),
                       actual=y)

# multi-statistic combination: predict log_lapvar from predicted (log_lapvar, reblur)
print("\n--- 2-stat linear blend (predict log_lapvar target) ---")
Xs = np.column_stack([results[s]["pred_both"] for s in STATS])
# blend weights fitted by LOO on train_sub ONLY (never on the holes)
Xt, yt = [], []
for st in STATS:
    obs = {f: T[f][st] for f in SUB}
    bp = BlurPredictor.fit(obs, mode="ll")
    Xt.append(bp.loo())
Xt = np.column_stack(Xt)
yt = np.array([T[f]["log_lapvar"] for f in SUB])
A = np.column_stack([np.ones(len(Xt)), Xt])
coef, *_ = np.linalg.lstsq(A, yt, rcond=None)
pb = np.column_stack([np.ones(len(EVAL)), Xs]) @ coef
print("  blend coef (1,log_lapvar,log_hf025,reblur) =", np.round(coef, 4))
print(f"  R2 on 28 holes (target log_lapvar): {r2(results['log_lapvar']['actual'], pb):.3f} "
      f"vs single-stat {r2(results['log_lapvar']['actual'], results['log_lapvar']['pred_both']):.3f}")

# ============================================================= STEP 3
print("\n" + "=" * 96)
print("STEP 3  -- the render-relative quantity:  log(GT sharp) - log(render sharp)")
print("=" * 96)
for st, rst in [("log_lapvar", "log_lapvar"), ("log_hf025", "log_hf025")]:
    gt = np.array([T[f][st] for f in EVAL])
    rn = np.array([R[f][rst] for f in EVAL])
    ratio = gt - rn                                  # log ratio
    pred_ratio_b = results[st]["pred_both"] - rn     # render sharpness is KNOWN at test time
    pred_ratio_o = results[st]["pred_one"] - rn
    print(f"\n  {st}: GT sd {gt.std(ddof=1):.4f}  render sd {rn.std(ddof=1):.4f}  "
          f"logratio mean {ratio.mean():+.4f} (= x{np.exp(ratio.mean()):.3f}) sd {ratio.std(ddof=1):.4f}")
    print(f"    corr(GT, render) = {np.corrcoef(gt, rn)[0,1]:+.3f}")
    print(f"    R2 of legal predictor on the LOG-RATIO: both-nb {r2(ratio, pred_ratio_b):.3f} "
          f" one-nb {r2(ratio, pred_ratio_o):.3f}")
    print(f"    RMSE(log-ratio) {rmse(ratio, pred_ratio_b):.4f}  "
          f"(a constant-offset baseline would give RMSE {ratio.std():.4f})")
for st in ["reblur"]:
    gt = np.array([T[f][st] for f in EVAL]); rn = np.array([R[f][st] for f in EVAL])
    ratio = gt - rn
    pb_ = results[st]["pred_both"] - rn
    print(f"\n  {st} (difference, not ratio): GT mean {gt.mean():.4f} render mean {rn.mean():.4f} "
          f"diff sd {ratio.std(ddof=1):.4f}  R2 {r2(ratio, pb_):.3f}")

np.save(f"{OUT}/_cache.npy", np.array([0]))

# ============================================================= STEP 4
pf = f"{OUT}/perframe_sr01.csv"
if os.path.exists(pf):
    print("\n" + "=" * 96)
    print("STEP 4  -- does predicted sharpness explain the per-frame baseline score?")
    print("=" * 96)
    S = load_csv(pf)
    frames = EVAL
    sc = np.array([S[f]["score"] for f in frames])
    ps = np.array([S[f]["psnr"] for f in frames])
    ss = np.array([S[f]["ssim"] for f in frames])
    lp = np.array([S[f]["lpips"] for f in frames])
    from scipy.stats import spearmanr
    print(f"  per-frame score: mean {sc.mean():.3f} sd {sc.std(ddof=1):.3f} "
          f"min {sc.min():.3f} max {sc.max():.3f}")
    for name, x in [("ACTUAL log_lapvar", np.array([T[f]['log_lapvar'] for f in frames])),
                    ("PRED  log_lapvar", results['log_lapvar']['pred_both']),
                    ("PRED  log_lapvar (1nb)", results['log_lapvar']['pred_one']),
                    ("ACTUAL reblur", np.array([T[f]['reblur'] for f in frames])),
                    ("PRED  reblur", results['reblur']['pred_both']),
                    ("ACTUAL log_hf025", np.array([T[f]['log_hf025'] for f in frames])),
                    ("PRED  log_hf025", results['log_hf025']['pred_both']),
                    ("RENDER log_lapvar", np.array([R[f]['log_lapvar'] for f in frames])),
                    ("ACTUAL log-ratio", np.array([T[f]['log_lapvar'] - R[f]['log_lapvar'] for f in frames])),
                    ("PRED  log-ratio", results['log_lapvar']['pred_both']
                     - np.array([R[f]['log_lapvar'] for f in frames]))]:
        out = [f"  {name:24s}"]
        for mn, m in [("score", sc), ("psnr", ps), ("ssim", ss), ("lpips", lp)]:
            r = np.corrcoef(x, m)[0, 1]
            rs = spearmanr(x, m).statistic
            out.append(f"{mn} r={r:+.3f}/rho={rs:+.3f}")
        print("  ".join(out))
    # regression score ~ predicted sharpness
    x = results['log_lapvar']['pred_both']
    b = np.polyfit(x, sc, 1)
    resid = sc - np.polyval(b, x)
    print(f"\n  score ~ PRED log_lapvar: slope {b[0]:+.3f} pts/nat, R2={r2(sc, np.polyval(b,x)):.3f}, "
          f"resid sd {resid.std(ddof=1):.3f} pts (total sd {sc.std(ddof=1):.3f})")
    xa = np.array([T[f]['log_lapvar'] for f in frames])
    ba = np.polyfit(xa, sc, 1)
    print(f"  score ~ ACTUAL log_lapvar: slope {ba[0]:+.3f} pts/nat, "
          f"R2={r2(sc, np.polyval(ba,xa)):.3f}")
    # write the joint table
    with open(f"{OUT}/holes_table.csv", "w") as fh:
        fh.write("frame,gt_log_lapvar,pred_log_lapvar_both,pred_log_lapvar_one,"
                 "gt_reblur,pred_reblur,render_log_lapvar,log_ratio,pred_log_ratio,"
                 "psnr,ssim,lpips,score\n")
        for i, f in enumerate(frames):
            fh.write(f"{f},{T[f]['log_lapvar']:.6f},{results['log_lapvar']['pred_both'][i]:.6f},"
                     f"{results['log_lapvar']['pred_one'][i]:.6f},{T[f]['reblur']:.6f},"
                     f"{results['reblur']['pred_both'][i]:.6f},{R[f]['log_lapvar']:.6f},"
                     f"{T[f]['log_lapvar']-R[f]['log_lapvar']:.6f},"
                     f"{results['log_lapvar']['pred_both'][i]-R[f]['log_lapvar']:.6f},"
                     f"{S[f]['psnr']:.4f},{S[f]['ssim']:.5f},{S[f]['lpips']:.5f},{S[f]['score']:.4f}\n")
    print(f"\n  wrote {OUT}/holes_table.csv")
else:
    print("\n(per-frame scores not ready yet)")

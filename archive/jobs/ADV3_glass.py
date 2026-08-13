import os, sys, numpy as np
from PIL import Image

GY, GX = 8, 12

def cellize(a, GY, GX):
    H, W = a.shape
    ys = np.linspace(0, H, GY+1).astype(int); xs = np.linspace(0, W, GX+1).astype(int)
    return np.array([[a[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() for j in range(GX)] for i in range(GY)])

def hf(g):
    gg = g.mean(2)
    return np.abs(np.diff(gg, axis=0, prepend=gg[:1])) + np.abs(np.diff(gg, axis=1, prepend=gg[:, :1]))

def run(gtdir, armdirs, tag, maxn=999):
    files = sorted(os.listdir(gtdir))[:maxn]
    E1 = np.zeros((GY, GX)); EM = np.zeros((GY, GX)); HFa = np.zeros((GY, GX))
    LUM = np.zeros((GY, GX)); ESPREAD = np.zeros((GY, GX)); n = 0
    for fn in files:
        st = os.path.splitext(fn)[0]
        g = np.asarray(Image.open(os.path.join(gtdir, fn)).convert("RGB"), np.float32)
        rs = []
        for ad in armdirs:
            p = os.path.join(ad, st + ".jpg")
            if not os.path.exists(p): p = os.path.join(ad, st + ".png")
            if not os.path.exists(p): continue
            r = np.asarray(Image.open(p).convert("RGB"), np.float32)
            if r.shape != g.shape: r = np.asarray(Image.open(p).convert("RGB").resize((g.shape[1], g.shape[0])), np.float32)
            rs.append(r)
        if len(rs) < 2: continue
        rs = np.stack(rs)
        e_each = np.abs(rs - g[None]).mean(3)                 # (A,H,W)
        E1 += cellize(e_each.mean(0), GY, GX)                 # mean single-arm error
        em = np.abs(rs.mean(0) - g).mean(2)
        EM += cellize(em, GY, GX)                             # ensemble-mean error
        HFa += cellize(hf(g), GY, GX)
        LUM += cellize(g.mean(2), GY, GX)
        ESPREAD += cellize(rs.std(0).mean(2), GY, GX)         # member disagreement
        n += 1
    return dict(tag=tag, n=n, E1=E1/n, EM=EM/n, HF=HFa/n, LUM=LUM/n, SPR=ESPREAD/n)

def report(d, rowgroups):
    print("\n===== %s  (n=%d views) =====" % (d["tag"], d["n"]))
    e, h = d["EM"].ravel(), d["HF"].ravel()
    e1 = d["E1"].ravel()
    # linear model with intercept
    for nm, y in (("single-arm-mean", e1), ("ensemble-mean", e)):
        b, a = np.polyfit(h, y, 1)
        pred = a + b*h
        res = (y - pred).reshape(GY, GX)
        rat = (y/h).reshape(GY, GX)
        print(" %-16s  fit err = %.3f + %.3f*hf   R2=%.3f" %
              (nm, a, b, np.corrcoef(pred, y)[0, 1]**2))
        for gname, rows in rowgroups:
            m = np.array(rows)
            print("   %-22s ratio=%.3f  resid_vs_fit=%+.3f/255 (%+.1f%% of local err)  hf=%.2f err=%.2f" %
                  (gname, y.reshape(GY, GX)[m].mean()/d["HF"][m].mean(),
                   res[m].mean(), 100*res[m].mean()/y.reshape(GY, GX)[m].mean(),
                   d["HF"][m].mean(), y.reshape(GY, GX)[m].mean()))
    # systematic fraction: ensemble error / single error ; higher = irreducible by ensembling
    sysf = d["EM"]/d["E1"]
    spr = d["SPR"]
    print("  systematic frac EM/E1 and member spread:")
    for gname, rows in rowgroups:
        m = np.array(rows)
        print("   %-22s EM/E1=%.4f   spread=%.3f  LUM=%.1f" % (gname, sysf[m].mean(), spr[m].mean(), d["LUM"][m].mean()))

BON_GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
BON_ARMS = sorted([os.path.join("/mnt/d/avv/bonsai_eval", a, "eval_render")
                   for a in os.listdir("/mnt/d/avv/bonsai_eval")])
CH_GT = "/mnt/d/avv/evalsplit/chair/eval_gt"
CH_ARMS = sorted([os.path.join("/mnt/d/avv/chair_eval", a, "eval_render")
                  for a in os.listdir("/mnt/d/avv/chair_eval")])

RG = [("top rows 0-1 (bg)", [0, 1]), ("mid rows 3-4", [3, 4]), ("bottom rows 5-7 (glass)", [5, 6, 7])]

db = run(BON_GT, BON_ARMS, "BONSAI 7-arm")
report(db, RG)
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV3_bonsai.npy", db, allow_pickle=True)
dc = run(CH_GT, CH_ARMS, "CHAIR 7-arm")
report(dc, RG)
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV3_chair.npy", dc, allow_pickle=True)
np.set_printoptions(precision=2, suppress=True, linewidth=220)
print("\nBONSAI ensemble-mean err per cell:"); print(db["EM"])
print("\nBONSAI GT luminance per cell:"); print(db["LUM"])
print("\nCHAIR ensemble-mean err per cell:"); print(dc["EM"])
print("\nCHAIR GT hf per cell:"); print(dc["HF"])

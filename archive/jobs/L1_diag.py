"""DIAGNOSTIC: how much sub-pixel jitter is there BETWEEN ensemble members?
If members are misregistered w.r.t. each other, the pixel-mean is blurred by that jitter and
align-then-merge (burst photography) recovers it. If jitter << 0.05px the lever is dead.
No GT is used to estimate anything here."""
import os, sys, time
import numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
import agg_lib as A

P7 = ["gsplatB9ut","gsplatB10ut8M","gsplatB11ut60k","gsplatB12ut8Ms7",
      "m31b_taillpips","e17visnorm","gsplatB8pure"]
stems, gt_by = A.stems_for(P7)
sub = stems[::10]     # 6 views
print("views:", len(sub))

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
dis.setFinestScale(0)
dis.setPatchSize(8); dis.setPatchStride(3)
dis.setUseMeanNormalization(True); dis.setUseSpatialPropagation(True)
dis.setVariationalRefinementIterations(5)

def gray(x): return (np.clip(x,0,1)*255).astype(np.uint8) @ np.array([0.114,0.587,0.299])  # BGR-ish irrelevant
def g8(x): return cv2.cvtColor((np.clip(x,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

rows=[]
t0=time.time()
for s in sub:
    X = np.stack([A.load(A.mdir(v), s) for v in P7])           # (k,H,W,3)
    M = X.mean(0)
    gm = g8(M)
    mags=[]; flows=[]
    for i in range(len(P7)):
        f = dis.calc(gm, g8(X[i]), None)                       # ref->member
        flows.append(f)
        mags.append(float(np.sqrt((f**2).sum(-1)).mean()))
    F = np.stack(flows)
    # residual jitter after removing the (near-zero) consensus flow
    fbar = F.mean(0)
    jit = np.sqrt(((F-fbar[None])**2).sum(-1))                 # (k,H,W)
    # global translation component per member (phase-correlation-like: mean of flow)
    gt_trans = [tuple(np.round(F[i].reshape(-1,2).mean(0),4)) for i in range(len(P7))]
    # sharpness proxy: gradient energy of mean vs of member0
    def ge(a):
        gx=np.diff(a,axis=1); gy=np.diff(a,axis=0)
        return float((gx**2).mean()+(gy**2).mean())
    rows.append((s, np.mean(mags), float(jit.mean()), float(np.median(jit)),
                 float(np.percentile(jit,90)), ge(M), np.mean([ge(X[i]) for i in range(len(P7))]),
                 gt_trans[:3]))
    print(f"{s[-12:]:14s} |flow| {np.mean(mags):.4f}px  jitter mean {jit.mean():.4f} med {np.median(jit):.4f} p90 {np.percentile(jit,90):.4f}"
          f"  gradE mean {ge(M):.5f} vs member {np.mean([ge(X[i]) for i in range(len(P7))]):.5f}"
          f"  ({100*(1-ge(M)/np.mean([ge(X[i]) for i in range(len(P7))])):.1f}% HF lost to averaging)")
print(f"\n{time.time()-t0:.0f}s")
J=[r[2] for r in rows]
print(f"MEAN inter-member jitter over views: {np.mean(J):.4f} px")
print("global translation of first 3 members (should be ~0 if only local jitter):")
for r in rows[:2]: print("  ", r[7])

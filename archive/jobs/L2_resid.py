"""LEVER 2 DIAGNOSTIC: TRANSFERABLE RESIDUAL from temporally-bracketing TRAIN views.

NOT the failed IBR photo-warp. We never move a GT photo into the test frame. We move the
RESIDUAL r = GT_train - render_train, which is the model's RECONSTRUCTION ERROR, and ask whether
the model makes the SAME error at the neighbouring test pose. Flow is estimated render->render
(same model, same appearance) so the correspondence problem is easy and photometrically clean.

If corr(r_test, warp(r_train)) = c, the best achievable MSE reduction is c^2 (alpha=c*sd ratio).
c<0.10 => <1% MSE => <0.05 dB => DEAD. c>0.30 => >9% MSE => >0.4 dB => worth building.
We also report the HIGH-PASS-ONLY correlation, because the low-frequency part of r is the lens
field + exposure, which production ALREADY exploits; only the incremental HF part would be new.
"""
import os, sys, re, json, time
import numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
import agg_lib as A

TRR = "/mnt/d/avv/output/HCM0181_gsplatB9ut/train_renders"
TRG = "/mnt/d/avv/data/phase1/public_set/HCM0181/train/images"
TER = "/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png"
TEG = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
idx = lambda f: int(re.search(r"_(\d{4})_V", f).group(1))

trs = {idx(f): os.path.splitext(f)[0] for f in os.listdir(TRR)}
trg = {idx(f): os.path.join(TRG, f) for f in os.listdir(TRG)}
tes = {idx(f): os.path.splitext(f)[0] for f in os.listdir(TEG)}
teg = {idx(f): os.path.join(TEG, f) for f in os.listdir(TEG)}
common_tr = sorted(set(trs) & set(trg))
print(f"train renders with GT: {len(common_tr)}   test views: {len(tes)}")

H, W = 989, 1320
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
dis.setUseSpatialPropagation(True); dis.setVariationalRefinementIterations(10)
g8 = lambda x: cv2.cvtColor((np.clip(x,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
ld  = A.load_gt

def hp(a, r=4):
    return a - cv2.GaussianBlur(a, (0,0), r)

sel = sorted(tes)[::3]
rows = []
t0 = time.time()
for ti in sel:
    rt = A.load(TER, tes[ti]); gtt = ld(teg[ti])
    r_test = gtt - rt
    j = min(common_tr, key=lambda c: abs(c-ti))
    rtr = A.load(TRR, trs[j]); gtr = ld(trg[j])
    r_tr = gtr - rtr
    F  = dis.calc(g8(rt), g8(rtr), None)                 # test -> train
    Fb = dis.calc(g8(rtr), g8(rt), None)
    mx, my = gx+F[...,0], gy+F[...,1]
    fb = np.sqrt((cv2.remap(Fb, mx, my, cv2.INTER_LINEAR) + F)**2).sum(-1)
    ok = (fb < 1.0) & (mx>1)&(mx<W-2)&(my>1)&(my<H-2)    # fwd-bwd consistent & inside
    wr = cv2.remap(r_tr, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    m = ok[...,None] & np.ones((1,1,3), bool)
    a, b = r_test[m], wr[m]
    c = float(np.corrcoef(a, b)[0,1])
    ah, bh = hp(r_test)[m], hp(wr)[m]
    ch = float(np.corrcoef(ah, bh)[0,1])
    rows.append((ti, j, abs(ti-j), float(ok.mean()), c, ch,
                 float(np.sqrt((F**2).sum(-1)).mean())))
    print(f"test {ti:4d} <- train {j:4d} (gap {abs(ti-j)})  valid {100*ok.mean():5.1f}%  "
          f"flow {np.sqrt((F**2).sum(-1)).mean():6.1f}px   corr(resid) {c:+.4f}   corr(HP resid) {ch:+.4f}", flush=True)

C  = np.array([r[4] for r in rows]); CH = np.array([r[5] for r in rows])
print(f"\n{time.time()-t0:.0f}s   n={len(rows)}")
print(f"  MEAN corr(residual)     = {C.mean():+.4f}  -> best-case MSE reduction {100*C.mean()**2:5.2f}%  = {-10*np.log10(1-C.mean()**2):+.3f} dB")
print(f"  MEAN corr(HP residual)  = {CH.mean():+.4f}  -> best-case MSE reduction {100*CH.mean()**2:5.2f}%  = {-10*np.log10(1-CH.mean()**2):+.3f} dB")
print(f"  (dScore from PSNR alone = 0.6 * dB)")
json.dump([list(map(float,r)) for r in rows], open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/L2_resid.json","w"))

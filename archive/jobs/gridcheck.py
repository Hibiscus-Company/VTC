import os, sys, glob, numpy as np, cv2, json

GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
ARMS = {
 "K4_pC_seed7(UT,classic - DEAD arm, 71.219)": "/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_png",
 "K1_noUT_aa (SHIPPED class, 71.90)"        : "/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png",
 "K4_pC_seed1k (UT, other seed)"            : "/mnt/d/avv/bonsai_eval/K4_pC_seed1k/eval_png",
}
R, C = 8, 12
names = sorted(os.listdir(GT))

def hp(x):
    return np.abs(x - cv2.GaussianBlur(x, (0,0), 2.0))

res = {}
for arm, rd in ARMS.items():
    if not os.path.isdir(rd):
        print("MISSING", rd); continue
    E = np.zeros((len(names), R, C)); H = np.zeros_like(E); L = np.zeros_like(E)
    Edm = np.zeros_like(E)   # error after removing per-image global mean offset
    for i, n in enumerate(names):
        g = cv2.imread(os.path.join(GT, n)).astype(np.float32)
        r = cv2.imread(os.path.join(rd, n.replace(".jpg",".png"))).astype(np.float32)
        assert g.shape == r.shape, (g.shape, r.shape, n)
        d = r - g
        off = d.reshape(-1,3).mean(0)          # per-image global colour offset (exposure drift)
        ad, adm = np.abs(d).mean(2), np.abs(d-off).mean(2)
        gg = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        hf = hp(gg); lum = gg
        H0, W0 = ad.shape
        for a in range(R):
            for b in range(C):
                sl = (slice(a*H0//R,(a+1)*H0//R), slice(b*W0//C,(b+1)*W0//C))
                E[i,a,b]=ad[sl].mean(); Edm[i,a,b]=adm[sl].mean()
                H[i,a,b]=hf[sl].mean(); L[i,a,b]=lum[sl].mean()
    res[arm]=dict(E=E,H=H,L=L,Edm=Edm)

def band(E,H,rows):
    e=E[:,rows,:].mean(); h=H[:,rows,:].mean(); return e,h,e/h

for arm,d in res.items():
    E,H,L,Edm = d["E"],d["H"],d["L"],d["Edm"]
    print("\n=== %s ==="%arm)
    for lab,rows in [("top rows0-1 (bg)",[0,1]),("mid rows3-4 (plant)",[3,4]),("bot rows5-7 (glass)",[5,6,7])]:
        e,h,rr=band(E,H,rows); em,_,rm=band(Edm,H,rows)
        print("  %-22s err %.3f  hf %.3f  ratio %.4f   | offset-removed err %.3f ratio %.4f"%(lab,e,h,rr,em,rm))
    # per-image bootstrap on the contrast bot-vs-mid
    pi_bot = E[:,[5,6,7],:].mean((1,2))/H[:,[5,6,7],:].mean((1,2))
    pi_mid = E[:,[3,4],:].mean((1,2))/H[:,[3,4],:].mean((1,2))
    dlt = pi_bot-pi_mid
    print("  bot-mid ratio delta: %.4f  +- %.4f (SEM over 28 imgs)  t=%.2f"%(dlt.mean(), dlt.std(ddof=1)/np.sqrt(len(dlt)), dlt.mean()/(dlt.std(ddof=1)/np.sqrt(len(dlt)))))
    pim_bot = Edm[:,[5,6,7],:].mean((1,2))/H[:,[5,6,7],:].mean((1,2))
    pim_mid = Edm[:,[3,4],:].mean((1,2))/H[:,[3,4],:].mean((1,2))
    dm = pim_bot-pim_mid
    print("  offset-removed delta: %.4f  t=%.2f"%(dm.mean(), dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm)))))

# ---- material vs near-field discriminator, on BOTTOM ROWS ONLY ----
# glossy black glass = DARK cells. carpet/pot/other near-field content = brighter.
print("\n### BOTTOM-ROW SPLIT BY GT LUMINANCE (material test) ###")
for arm,d in res.items():
    E,H,L = d["E"],d["H"],d["L"]
    eb=E[:,5:,:].ravel(); hb=H[:,5:,:].ravel(); lb=L[:,5:,:].ravel()
    q=np.quantile(lb,[0.33,0.67])
    for lab,m in [("darkest 3rd (BLACK GLASS)",lb<=q[0]),("mid 3rd",(lb>q[0])&(lb<=q[1])),("brightest 3rd",lb>q[1])]:
        print("  %-28s %-38s lum %.1f  err %.3f hf %.3f ratio %.4f  n=%d"%(arm.split("(")[0],lab,lb[m].mean(),eb[m].mean(),hb[m].mean(),eb[m].mean()/hb[m].mean(),m.sum()))

# ---- full per-row profile: is it a monotone top->bottom (near-field) gradient? ----
print("\n### PER-ROW ratio profile (near-field gradient test) ###")
for arm,d in res.items():
    E,H=d["E"],d["H"]
    print("  %-14s "%arm.split("(")[0] + " ".join("r%d %.3f"%(a,E[:,a,:].mean()/H[:,a,:].mean()) for a in range(R)))

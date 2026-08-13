import os, numpy as np, cv2
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"
RD="/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png"   # SHIPPED-class arm
R,C=8,12
names=sorted(os.listdir(GT))
def hp(x): return np.abs(x-cv2.GaussianBlur(x,(0,0),2.0))

E0=np.zeros((len(names),R,C)); ES=np.zeros_like(E0); HG=np.zeros_like(E0); HR=np.zeros_like(E0); LU=np.zeros_like(E0); SH=np.zeros_like(E0)
for i,n in enumerate(names):
    g=cv2.imread(os.path.join(GT,n)).astype(np.float32)
    r=cv2.imread(os.path.join(RD,n.replace(".jpg",".png"))).astype(np.float32)
    gg=cv2.cvtColor(g,cv2.COLOR_BGR2GRAY); rg=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    hg=hp(gg); hr=hp(rg); H0,W0=gg.shape
    for a in range(R):
        for b in range(C):
            y0,y1=a*H0//R,(a+1)*H0//R; x0,x1=b*W0//C,(b+1)*W0//C
            P=4
            yy0,yy1=max(y0-P,0),min(y1+P,H0); xx0,xx1=max(x0-P,0),min(x1+P,W0)
            gt=g[y0:y1,x0:x1]; big=r[yy0:yy1,xx0:xx1]
            base=np.abs(r[y0:y1,x0:x1]-gt).mean()
            best=base; bs=0.0
            for dy in range(-3,4):
                for dx in range(-3,4):
                    sy,sx=y0-yy0+dy, x0-xx0+dx
                    if sy<0 or sx<0 or sy+(y1-y0)>big.shape[0] or sx+(x1-x0)>big.shape[1]: continue
                    v=np.abs(big[sy:sy+(y1-y0),sx:sx+(x1-x0)]-gt).mean()
                    if v<best: best=v; bs=np.hypot(dy,dx)
            E0[i,a,b]=base; ES[i,a,b]=best; SH[i,a,b]=bs
            HG[i,a,b]=hg[y0:y1,x0:x1].mean(); HR[i,a,b]=hr[y0:y1,x0:x1].mean(); LU[i,a,b]=gg[y0:y1,x0:x1].mean()

def bnd(rows,lab):
    e0=E0[:,rows,:].mean(); es=ES[:,rows,:].mean(); hg=HG[:,rows,:].mean(); hr=HR[:,rows,:].mean()
    print("  %-20s err %.3f -> best-shift %.3f (removed %4.1f%%)  mean|shift| %.2fpx | render_hf/GT_hf %.3f | ratio_raw %.3f ratio_aftershift %.3f"%(
        lab,e0,es,100*(1-es/e0),SH[:,rows,:].mean(),hr/hg,e0/hg,es/hg))
print("### MISREGISTRATION vs VIEW-DEPENDENCE (K1_noUT_aa, n=28, +-3px local shift) ###")
bnd([0,1],"top rows0-1 bg"); bnd([3,4],"mid rows3-4 plant"); bnd([5,6,7],"bot rows5-7 glass")
print("\n### bottom rows, luminance terciles ###")
eb0=E0[:,5:,:].ravel(); ebs=ES[:,5:,:].ravel(); hb=HG[:,5:,:].ravel(); hrb=HR[:,5:,:].ravel(); lb=LU[:,5:,:].ravel(); sb=SH[:,5:,:].ravel()
q=np.quantile(lb,[0.33,0.67])
for lab,m in [("darkest (BLACK GLASS)",lb<=q[0]),("mid",(lb>q[0])&(lb<=q[1])),("brightest",lb>q[1])]:
    print("  %-22s lum %5.1f  ratio_raw %.3f  ratio_aftershift %.3f  removed %4.1f%%  shift %.2fpx  rHF/gHF %.3f"%(
        lab,lb[m].mean(),eb0[m].mean()/hb[m].mean(),ebs[m].mean()/hb[m].mean(),100*(1-ebs[m].mean()/eb0[m].mean()),sb[m].mean(),hrb[m].mean()/hb[m].mean()))
print("\n### per-row: raw ratio / after-shift ratio / shift px ###")
for a in range(R):
    print("  r%d  raw %.3f  shifted %.3f  removed %4.1f%%  shift %.2fpx  rHF/gHF %.3f"%(
        a,E0[:,a,:].mean()/HG[:,a,:].mean(),ES[:,a,:].mean()/HG[:,a,:].mean(),
        100*(1-ES[:,a,:].mean()/E0[:,a,:].mean()),SH[:,a,:].mean(),HR[:,a,:].mean()/HG[:,a,:].mean()))

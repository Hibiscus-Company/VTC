import numpy as np,sys
A=np.load(sys.argv[1]); cnt,err,grad=A[:,0],A[:,1],A[:,2]
rng=np.random.default_rng(0)
def oracle(cntv,nq=4):
    qe=np.quantile(grad,np.linspace(0,1,nq+1)); qi=np.digitize(grad,qe[1:-1])
    tot=err.sum(); saved=0.
    for q in range(nq):
        m=qi==q; c=cntv[m]; e=err[m]
        t1,t2=np.quantile(c,[1/3,2/3]); sp=c<=t1; dn=c>=t2
        if sp.sum()<5 or dn.sum()<5: continue
        a,b=e[sp].mean(),e[dn].mean()
        if a>b: saved+=e[sp].sum()*(1-b/a)
    base=-10*np.log10(err.mean()); return -10*np.log10((tot-saved)/len(err))-base
print('REAL   oracle %+.4f dB'%oracle(cnt))
for nq in (4,):
    ps=[]
    qe=np.quantile(grad,np.linspace(0,1,nq+1)); qi=np.digitize(grad,qe[1:-1])
    for it in range(40):
        c2=cnt.copy()
        for q in range(nq):
            m=np.where(qi==q)[0]; c2[m]=rng.permutation(cnt[m])
        ps.append(oracle(c2,nq))
    ps=np.array(ps)
    print('PLACEBO (density shuffled within detail bin, n=40): mean %+.4f  sd %.4f  max %+.4f'%(ps.mean(),ps.std(),ps.max()))
# placebo 2: an unrelated real covariate = cell index parity-free random gaussian
ps2=[oracle(rng.standard_normal(len(cnt))) for _ in range(20)]
print('PLACEBO2 (pure random covariate, n=20): mean %+.4f  sd %.4f'%(np.mean(ps2),np.std(ps2)))

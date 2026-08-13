import os,sys,torch,numpy as np
SC={"K1_noUT_aa":71.9029,"K4_pC_seed1k":71.3067,"K4_pC_seed7":71.2190,"eps10":71.5985,
    "eps20":71.5260,"ppisp_pc":71.2285,"K2_clip_full":None}
rows=[]
for a,s in SC.items():
    p=f"/mnt/d/avv/bonsai_eval/{a}/ckpt.pt"
    if not os.path.exists(p): continue
    c=torch.load(p,map_location="cpu",weights_only=False); sp=c["splats"]
    S=torch.exp(sp["scales"]); o=torch.sigmoid(sp["opacities"])
    tot=len(S); m=(o>0.05)
    Sx=S[m].double().numpy(); Ss=np.sort(Sx,axis=1)[:,::-1]
    r=dict(arm=a,score=s,total=tot,N=int(m.sum()),frac=float(m.float().mean()),
           s2s1=float(np.median(Ss[:,1]/Ss[:,0])),
           s2s3=float(np.median(Ss[:,1]/np.maximum(Ss[:,2],1e-30))),
           s1s3=float(np.median(Ss[:,0]/np.maximum(Ss[:,2],1e-30))),
           medop=float(o.median()), meanop=float(o.mean()),
           vol=float(np.median(Sx.prod(1))))
    rows.append(r)
    print(f"{a:14s} score={str(s):>8s} total={tot:>9,} N(op>.05)={r['N']:>9,} ({100*r['frac']:5.2f}%) "
          f"s2/s1 {r['s2s1']:.4f} s2/s3 {r['s2s3']:>10.1f} s1/s3 {r['s1s3']:>10.1f}",flush=True)
    del c,sp,S,o,Sx,Ss
import json; json.dump(rows,open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/CENSUS_all.json","w"))
z=[r for r in rows if r["score"] is not None]
from scipy.stats import spearmanr,pearsonr
sc=np.array([r["score"] for r in z])
for k in ["N","s2s1","s2s3","s1s3","medop","vol"]:
    v=np.array([r[k] for r in z])
    print(f"  {k:6s} vs score: spearman {spearmanr(v,sc).statistic:+.3f}  pearson {pearsonr(v,sc).statistic:+.3f}  (n={len(z)})")

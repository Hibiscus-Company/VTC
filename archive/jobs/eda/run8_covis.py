"""Co-visibility: number of AVAILABLE train images sharing >=T 3D points with the query view.
This is the strongest single predictor of NVS quality at a held-out view, and it captures
BOTH local spacing and global training-set size. Real test vs eval-split proxy, like-for-like."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
from colmap_io import read_images_binary

def covis(src, train_names, query_names, thresholds=(50,100,300)):
    sp = find_sparse(src)
    imgs = read_images_binary(os.path.join(sp,"images.bin"))
    byname = {v["name"]: v for v in imgs.values()}
    tsets = {n: set(int(x) for x in byname[n]["point3D_ids"] if x>=0) for n in train_names if n in byname}
    res = {t: [] for t in thresholds}
    for q in query_names:
        if q not in byname: continue
        qs = set(int(x) for x in byname[q]["point3D_ids"] if x>=0)
        sh = np.array([len(qs & s) for s in tsets.values()])
        for t in thresholds:
            res[t].append(int((sh>=t).sum()))
    return {t: np.array(v) for t,v in res.items()}

def ball_density(Ctr, Cq, frac_of_radius=0.15):
    rad = np.linalg.norm(Ctr-Ctr.mean(0),axis=1).mean()
    R = frac_of_radius*rad
    D = np.linalg.norm(Cq[:,None,:]-Ctr[None,:,:],axis=2)
    return (D<=R).sum(1), R

rows=[]
print(f"{'case':22s} {'ntr':>4s} {'nq':>3s} | covis>=50 med  | covis>=100 med | covis>=300 med | ball15% med | d1/sp med")
print("-"*112)
for s in ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]:
    src=os.path.join(SET2,s)
    trn=sorted(os.listdir(os.path.join(src,"train/images")))
    te=load_poses_csv(os.path.join(src,"test/test_poses.csv")); ten=[t["name"] for t in te]
    c=covis(src,trn,ten)
    all_p,_=train_poses(src); tr=[p for p in all_p if p["name"] in set(trn)]
    Ctr=centers(tr); Cte=centers(te)
    bd,R=ball_density(Ctr,Cte)
    sp_=np.median(train_spacing(Ctr)); d1,_,_=nn_stats(Ctr,Cte)
    print(f"{'TEST  '+s:22s} {len(trn):4d} {len(ten):3d} | {np.median(c[50]):6.0f} p10={np.percentile(c[50],10):5.0f} | {np.median(c[100]):6.0f} p10={np.percentile(c[100],10):5.0f} | {np.median(c[300]):6.0f} p10={np.percentile(c[300],10):5.0f} | {np.median(bd):7.1f} | {np.median(d1)/sp_:.2f}")
    rows.append(("test",s,len(trn),float(np.median(c[100])),float(np.median(c[300])),float(np.median(bd))))
print()
for s in ["HCM0181","HCM0421","chair","bonsai"]:
    root=os.path.join(EVAL,s); cfg=json.load(open(os.path.join(root,"split.json")))
    src=cfg["scene"].replace("/home/bkai/data","/mnt/d/avv/data")
    sub=sorted(os.listdir(os.path.join(root,"train_sub/images")))
    ev=load_poses_csv(os.path.join(root,"eval_poses.csv")); evn=[e["name"] for e in ev]
    c=covis(src,sub,evn)
    all_p,_=train_poses(src); tr=[p for p in all_p if p["name"] in set(sub)]
    Ctr=centers(tr); Cev=centers(ev)
    bd,R=ball_density(Ctr,Cev)
    sp_=np.median(train_spacing(Ctr)); d1,_,_=nn_stats(Ctr,Cev)
    print(f"{'PROXY '+s:22s} {len(sub):4d} {len(evn):3d} | {np.median(c[50]):6.0f} p10={np.percentile(c[50],10):5.0f} | {np.median(c[100]):6.0f} p10={np.percentile(c[100],10):5.0f} | {np.median(c[300]):6.0f} p10={np.percentile(c[300],10):5.0f} | {np.median(bd):7.1f} | {np.median(d1)/sp_:.2f}")
    rows.append(("proxy",s,len(sub),float(np.median(c[100])),float(np.median(c[300])),float(np.median(bd))))
# public HCM0181 real test for reference
src=os.path.join(PUB,"HCM0181")
trn=sorted(os.listdir(os.path.join(src,"train/images")))
te=load_poses_csv(os.path.join(src,"test/test_poses.csv")); ten=[t["name"] for t in te]
c=covis(src,trn,ten)
all_p,_=train_poses(src); tr=[p for p in all_p if p["name"] in set(trn)]
Ctr=centers(tr); Cte=centers(te); bd,R=ball_density(Ctr,Cte)
sp_=np.median(train_spacing(Ctr)); d1,_,_=nn_stats(Ctr,Cte)
print(f"{'PUBTEST HCM0181':22s} {len(trn):4d} {len(ten):3d} | {np.median(c[50]):6.0f} p10={np.percentile(c[50],10):5.0f} | {np.median(c[100]):6.0f} p10={np.percentile(c[100],10):5.0f} | {np.median(c[300]):6.0f} p10={np.percentile(c[300],10):5.0f} | {np.median(bd):7.1f} | {np.median(d1)/sp_:.2f}")
json.dump(rows, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_covis.json","w"), indent=1)

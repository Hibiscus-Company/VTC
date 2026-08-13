"""SfM-observation support: for each query view (test or eval-hole), how much of what it
sees is also seen by the AVAILABLE training views? This is the content-level analogue of
pose distance and is the fairest proxy-vs-test comparison."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
from colmap_io import read_images_binary, read_points3D_binary

SCENES = ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]

def support(src, train_names, query_names):
    sp = find_sparse(src)
    imgs = read_images_binary(os.path.join(sp,"images.bin"))
    byname = {v["name"]: v for v in imgs.values()}
    # point -> set of train images observing it
    ptcount = {}
    for n in train_names:
        if n not in byname: continue
        for p in byname[n]["point3D_ids"]:
            if p >= 0:
                ptcount[p] = ptcount.get(p, 0) + 1
    fracs, ns, obs_per_pt = [], [], []
    for n in query_names:
        v = byname.get(n)
        if v is None: continue
        pts = v["point3D_ids"]; pts = pts[pts >= 0]
        if len(pts) == 0:
            fracs.append(0.0); ns.append(0); continue
        cnt = np.array([ptcount.get(int(p), 0) for p in pts])
        fracs.append(float((cnt > 0).mean()))
        ns.append(len(pts))
        obs_per_pt.append(float(np.median(cnt)))
    return np.array(fracs), np.array(ns), np.array(obs_per_pt), byname, ptcount

def parallax(src, train_names, query_names, byname, ptcount, sample=25):
    """min triangulation angle between query view and nearest train view observing same point."""
    sp = find_sparse(src)
    ids, xyz, rgb, err, tl = read_points3D_binary(os.path.join(sp,"points3D.bin"))
    P = {int(i): x for i, x in zip(ids, xyz)}
    # train obs per point
    tobs = {}
    for n in train_names:
        v = byname.get(n)
        if v is None: continue
        c = cam_center(v["qvec"], v["tvec"])
        for p in v["point3D_ids"]:
            if p >= 0: tobs.setdefault(int(p), []).append(c)
    out = []
    for n in query_names:
        v = byname.get(n)
        if v is None: continue
        cq = cam_center(v["qvec"], v["tvec"])
        pts = v["point3D_ids"]; pts = pts[pts>=0]
        if len(pts)==0: continue
        sel = pts if len(pts)<=200 else np.random.default_rng(0).choice(pts, 200, replace=False)
        angs=[]
        for p in sel:
            p=int(p)
            if p not in tobs or p not in P: continue
            X = P[p]; d0 = cq - X; d0/= np.linalg.norm(d0)+1e-12
            C = np.array(tobs[p]); D = C - X; D/= (np.linalg.norm(D,axis=1,keepdims=True)+1e-12)
            a = np.degrees(np.arccos(np.clip(D@d0,-1,1)))
            angs.append(a.min())
        if angs: out.append(float(np.median(angs)))
    return np.array(out)

print("###### SET2 REAL TEST ######")
RES={}
for s in SCENES:
    src = os.path.join(SET2,s)
    trn = sorted(os.listdir(os.path.join(src,"train/images")))
    te = load_poses_csv(os.path.join(src,"test/test_poses.csv"))
    ten=[t["name"] for t in te]
    f, ns, opp, byname, ptcount = support(src, trn, ten)
    # baseline: train views' own support (leave-one-out: subtract self)
    ftr=[]
    for n in trn:
        v=byname[n]; pts=v["point3D_ids"]; pts=pts[pts>=0]
        cnt=np.array([ptcount.get(int(p),0) for p in pts])
        ftr.append(float((cnt>1).mean()))   # >1 excludes self
    par = parallax(src, trn, ten, byname, ptcount)
    RES[s]=dict(test_supp_med=float(np.median(f)), test_supp_p10=pct(f,10), test_supp_min=float(f.min()),
                train_loo_supp_med=float(np.median(ftr)),
                test_npts_med=float(np.median(ns)),
                test_parallax_med=float(np.median(par)), test_parallax_p90=pct(par,90))
    print(f"{s:9s} test supported-frac med={np.median(f):.3f} p10={pct(f,10):.3f} min={f.min():.3f} | trainLOO {np.median(ftr):.3f} | npts med={np.median(ns):.0f} | min-parallax-to-train med={np.median(par):.2f}deg p90={pct(par,90):.2f}")

print("\n###### EVAL-SPLIT PROXIES ######")
PROX=["HCM0181","HCM0421","chair","bonsai"]
for s in PROX:
    root=os.path.join(EVAL,s)
    cfg=json.load(open(os.path.join(root,"split.json")))
    src=cfg["scene"].replace("/home/bkai/data","/mnt/d/avv/data")
    sub=sorted(os.listdir(os.path.join(root,"train_sub/images")))
    ev=load_poses_csv(os.path.join(root,"eval_poses.csv")); evn=[e["name"] for e in ev]
    f, ns, opp, byname, ptcount = support(src, sub, evn)
    ftr=[]
    for n in sub:
        v=byname[n]; pts=v["point3D_ids"]; pts=pts[pts>=0]
        cnt=np.array([ptcount.get(int(p),0) for p in pts]); ftr.append(float((cnt>1).mean()))
    par = parallax(src, sub, evn, byname, ptcount)
    RES["proxy_"+s]=dict(eval_supp_med=float(np.median(f)), eval_supp_p10=pct(f,10),
                train_loo_supp_med=float(np.median(ftr)), eval_npts_med=float(np.median(ns)),
                eval_parallax_med=float(np.median(par)), eval_parallax_p90=pct(par,90))
    print(f"{s:9s} eval supported-frac med={np.median(f):.3f} p10={pct(f,10):.3f} min={f.min():.3f} | trainLOO {np.median(ftr):.3f} | npts med={np.median(ns):.0f} | min-parallax-to-trainsub med={np.median(par):.2f}deg p90={pct(par,90):.2f}")

json.dump(RES, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_support.json","w"), indent=1)

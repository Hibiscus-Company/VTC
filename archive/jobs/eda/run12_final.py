import os, sys, json, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *

def tailstats(Ctr,Cq):
    sp=np.median(train_spacing(Ctr)); d1,dk,_=nn_stats(Ctr,Cq,4)
    n=d1/sp
    return dict(med=float(np.median(n)),p90=float(np.percentile(n,90)),max=float(n.max()),
                f_gt2=float((n>2).mean()),f_gt3=float((n>3).mean()),f_gt5=float((n>5).mean()),
                dk_med=float(np.median(dk/sp)),dk_p90=float(np.percentile(dk/sp,90)))

def runs_of(idx):
    idx=sorted(idx); r=[];c=1
    for a,b in zip(idx,idx[1:]):
        if b==a+1: c+=1
        else: r.append(c);c=1
    r.append(c); return r

print("### TAIL CALIBRATION: real TEST vs eval-split PROXY (d1 normalised by train NN-spacing)")
print(f"{'case':24s} {'ntrain':>6s} {'med':>6s} {'p90':>6s} {'max':>6s} {'%>2':>6s} {'%>3':>6s} {'%>5':>6s} {'dk4med':>7s}")
for s in ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]:
    src=os.path.join(SET2,s)
    all_p,_=train_poses(src); od=set(os.listdir(os.path.join(src,"train/images")))
    tr=[p for p in all_p if p["name"] in od]
    te=load_poses_csv(os.path.join(src,"test/test_poses.csv"))
    t=tailstats(centers(tr),centers(te))
    print(f"{'TEST  '+s:24s} {len(tr):6d} {t['med']:6.2f} {t['p90']:6.2f} {t['max']:6.2f} {t['f_gt2']*100:5.1f}% {t['f_gt3']*100:5.1f}% {t['f_gt5']*100:5.1f}% {t['dk_med']:7.2f}")
print()
for s in ["HCM0181","HCM0421","chair","bonsai"]:
    root=os.path.join(EVAL,s); cfg=json.load(open(os.path.join(root,"split.json")))
    src=cfg["scene"].replace("/home/bkai/data","/mnt/d/avv/data")
    sub=set(os.listdir(os.path.join(root,"train_sub/images")))
    all_p,_=train_poses(src); tr=[p for p in all_p if p["name"] in sub]
    ev=load_poses_csv(os.path.join(root,"eval_poses.csv"))
    t=tailstats(centers(tr),centers(ev))
    print(f"{'PROXY '+s:24s} {len(tr):6d} {t['med']:6.2f} {t['p90']:6.2f} {t['max']:6.2f} {t['f_gt2']*100:5.1f}% {t['f_gt3']*100:5.1f}% {t['f_gt5']*100:5.1f}% {t['dk_med']:7.2f}")
src=os.path.join(PUB,"HCM0181")
all_p,_=train_poses(src); od=set(os.listdir(os.path.join(src,"train/images")))
tr=[p for p in all_p if p["name"] in od]; te=load_poses_csv(os.path.join(src,"test/test_poses.csv"))
t=tailstats(centers(tr),centers(te))
print(f"{'PUBTEST HCM0181':24s} {len(tr):6d} {t['med']:6.2f} {t['p90']:6.2f} {t['max']:6.2f} {t['f_gt2']*100:5.1f}% {t['f_gt3']*100:5.1f}% {t['f_gt5']*100:5.1f}% {t['dk_med']:7.2f}")

print("\n### TRAINING-DATA BUDGET: proxy vs production")
print(f"{'scene':12s} {'prod_train':>10s} {'proxy_train':>11s} {'ratio':>6s} {'prod_sp':>8s} {'proxy_sp':>9s} {'sp_infl':>8s}")
pairs=[("HCM0181",os.path.join(PUB,"HCM0181"),os.path.join(EVAL,"HCM0181")),
       ("HCM0421",os.path.join(SET2,"HCM0421"),os.path.join(EVAL,"HCM0421")),
       ("chair",os.path.join(SET2,"chair"),os.path.join(EVAL,"chair")),
       ("bonsai",os.path.join(SET2,"bonsai"),os.path.join(EVAL,"bonsai"))]
for name,src,pr in pairs:
    all_p,_=train_poses(src)
    od=set(os.listdir(os.path.join(src,"train/images"))); sub=set(os.listdir(os.path.join(pr,"train_sub/images")))
    Cf=centers([p for p in all_p if p["name"] in od]); Cs=centers([p for p in all_p if p["name"] in sub])
    spf=np.median(train_spacing(Cf)); sps=np.median(train_spacing(Cs))
    print(f"{name:12s} {len(od):10d} {len(sub):11d} {len(sub)/len(od):6.3f} {spf:8.4f} {sps:9.4f} {sps/spf:7.3f}x")

print("\n### HOLD-OUT RUN STRUCTURE (consecutive held-out frames)")
def idx_of(n):
    m=re.search(r"_(\d{4})_V",n)
    if m: return int(m.group(1))
    m=re.search(r"frame_(\d+)",n)
    return int(m.group(1)) if m else None
for s in ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"]:
    src=os.path.join(SET2,s)
    te=load_poses_csv(os.path.join(src,"test/test_poses.csv"))
    r=runs_of([idx_of(t["name"]) for t in te])
    print(f"TEST  {s:10s} runs: max={max(r)} mean={np.mean(r):.2f} n_runs={len(r)} dist={dict(zip(*np.unique(r,return_counts=True)))}")
for s in ["HCM0181","HCM0421"]:
    root=os.path.join(EVAL,s); ev=load_poses_csv(os.path.join(root,"eval_poses.csv"))
    r=runs_of([idx_of(e["name"]) for e in ev])
    print(f"PROXY {s:10s} runs: max={max(r)} mean={np.mean(r):.2f} n_runs={len(r)} dist={dict(zip(*np.unique(r,return_counts=True)))}")

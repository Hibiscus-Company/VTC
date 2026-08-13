import os, sys, re, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *

SCENES = ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]
def idx_of(n):
    m = re.search(r"_(\d{4})_V", n)
    if m: return int(m.group(1))
    m = re.search(r"frame_(\d+)", n)
    if m: return int(m.group(1))
    return None
def ts_of(n):
    m = re.search(r"DJI_(\d{14})_", n)
    return m.group(1) if m else None

for s in SCENES:
    root = os.path.join(SET2, s)
    all_p,_ = train_poses(root)
    on_disk = set(os.listdir(os.path.join(root,"train/images")))
    te = load_poses_csv(os.path.join(root,"test/test_poses.csv"))
    te_names = set(t["name"] for t in te)
    tr_i = sorted(idx_of(n) for n in on_disk)
    te_i = sorted(idx_of(t["name"]) for t in te)
    ex_i = sorted(idx_of(p["name"]) for p in all_p if p["name"] not in on_disk and p["name"] not in te_names)
    allidx = sorted(set(tr_i)|set(te_i)|set(ex_i))
    # for each test index, is it strictly between two train indices, and gap size
    tr_arr = np.array(tr_i)
    gaps = []
    for i in te_i:
        lo = tr_arr[tr_arr < i]; hi = tr_arr[tr_arr > i]
        gaps.append((i - lo.max() if len(lo) else -1, hi.min()-i if len(hi) else -1))
    gaps = np.array(gaps)
    interior = ((gaps[:,0]>0)&(gaps[:,1]>0)).mean()
    # runs: consecutive test indices
    runs=[];cur=1
    for a,b in zip(te_i, te_i[1:]):
        if b==a+1: cur+=1
        else: runs.append(cur); cur=1
    runs.append(cur)
    print(f"=== {s}")
    print(f"  train idx range [{tr_arr.min()},{tr_arr.max()}] n={len(tr_i)}; test [{min(te_i)},{max(te_i)}] n={len(te_i)}; extra n={len(ex_i)}")
    print(f"  test strictly interior to train index range: {interior*100:.0f}%")
    print(f"  test gap-to-prev-train: med={np.median(gaps[:,0]):.0f} max={gaps[:,0].max()}; gap-to-next: med={np.median(gaps[:,1]):.0f} max={gaps[:,1].max()}")
    print(f"  consecutive-test run lengths: max={max(runs)} mean={np.mean(runs):.2f} n_runs={len(runs)}")
    ts = [ts_of(t["name"]) for t in te]
    if ts[0]:
        tstr = sorted(ts_of(n) for n in on_disk)
        print(f"  train time span {tstr[0]}..{tstr[-1]}; test {min(ts)}..{max(ts)}")
    # index coverage: what fraction of the full [min,max] index grid is train
    print(f"  union coverage: {len(allidx)} of range {allidx[-1]-allidx[0]+1}; train frac of union={len(tr_i)/len(allidx):.2f}")
    print(f"  first 25 test idx: {te_i[:25]}")

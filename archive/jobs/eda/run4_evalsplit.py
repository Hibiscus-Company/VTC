import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *

PROX = ["HCM0181", "HCM0421", "chair", "bonsai", "bonsai2"]
OUT = {}
for s in PROX:
    root = os.path.join(EVAL, s)
    cfg = json.load(open(os.path.join(root, "split.json")))
    src = cfg["scene"].replace("/home/bkai/data", "/mnt/d/avv/data")
    # train_sub poses: get from source sparse model, filtered by train_sub/images listing
    all_p, sp = train_poses(src)
    sub_names = set(os.listdir(os.path.join(root, "train_sub/images")))
    tr = [p for p in all_p if p["name"] in sub_names]
    ev = load_poses_csv(os.path.join(root, "eval_poses.csv"))
    Ctr, Atr = centers(tr), axes(tr)
    Cev, Aev = centers(ev), axes(ev)
    sp_tr = train_spacing(Ctr); med_sp = float(np.median(sp_tr))
    d1, dk, D = nn_stats(Ctr, Cev, 4)
    Dtt = np.linalg.norm(Ctr[:,None,:]-Ctr[None,:,:],axis=2); np.fill_diagonal(Dtt, np.inf)
    Dtts = np.sort(Dtt,1); tr_d1, tr_dk = Dtts[:,0], Dtts[:,:4].mean(1)
    inside, outd = hull_test(Ctr, Cev)
    a1, akmax = angle_to_nn(Ctr, Atr, Cev, Aev, 4)
    idx = np.argsort(Dtt,1)
    a1t = np.array([np.degrees(np.arccos(np.clip(Atr[idx[i,0]] @ Atr[i],-1,1))) for i in range(len(Ctr))])
    # ALSO compare against FULL train reference spacing (what the real test enjoys)
    full_names = set(os.listdir(os.path.join(src, "train/images")))
    trf = [p for p in all_p if p["name"] in full_names]
    Ctrf = centers(trf)
    spf = train_spacing(Ctrf); med_spf = float(np.median(spf))
    d1f,_,_ = nn_stats(Ctrf, Cev, 4)
    OUT[s] = dict(src=src, n_train_sub=len(tr), n_train_full=len(trf), n_eval=len(ev),
        med_sub_spacing=med_sp, med_full_spacing=med_spf,
        eval_d1_norm=summ(d1/med_sp), eval_dk_norm=summ(dk/med_sp),
        trainsub_loo_d1_norm=summ(tr_d1/med_sp),
        eval_d1_norm_vs_FULL=summ(d1f/med_spf),
        eval_d1_abs=summ(d1), eval_d1_abs_vs_full=summ(d1f),
        frac_inside_hull=float(inside.mean()),
        ang_nn1=summ(a1), ang_nn4max=summ(akmax), trainsub_ang_nn1=summ(a1t))
    print(f"=== proxy {s}: n_sub={len(tr)} (full {len(trf)}) n_eval={len(ev)}  med_sp_sub={med_sp:.4f} med_sp_full={med_spf:.4f}")
    print(f"   eval d1/sp_sub  med={np.median(d1/med_sp):.2f} p90={pct(d1/med_sp,90):.2f} max={(d1/med_sp).max():.2f}")
    print(f"   trainsub LOO d1/sp med={np.median(tr_d1/med_sp):.2f} p90={pct(tr_d1/med_sp,90):.2f}")
    print(f"   ABS: eval->sub d1 med={np.median(d1):.4f} | eval->FULLtrain d1 med={np.median(d1f):.4f}  ratio={np.median(d1)/np.median(d1f):.2f}x")
    print(f"   hull={inside.mean()*100:.1f}%  ang_nn1 med={np.median(a1):.2f} p90={pct(a1,90):.2f} (subLOO {np.median(a1t):.2f})")
json.dump(OUT, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_geom_evalsplit.json","w"), indent=1)

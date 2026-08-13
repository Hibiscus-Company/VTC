import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *

SCENES = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674", "chair", "bonsai"]
OUT = {}

def analyse(tag, Ctr, Atr, Cte, Ate, tr_names=None, te_names=None):
    sp_tr = train_spacing(Ctr); med_sp = float(np.median(sp_tr))
    d1, dk, D = nn_stats(Ctr, Cte, 4)
    Dtt = np.linalg.norm(Ctr[:, None, :] - Ctr[None, :, :], axis=2); np.fill_diagonal(Dtt, np.inf)
    Dtts = np.sort(Dtt, axis=1); tr_d1, tr_dk = Dtts[:, 0], Dtts[:, :4].mean(1)
    inside, outd = hull_test(Ctr, Cte)
    a1, akmax = angle_to_nn(Ctr, Atr, Cte, Ate, 4)
    idx = np.argsort(Dtt, axis=1)
    a1t = np.array([np.degrees(np.arccos(np.clip(Atr[idx[i,0]] @ Atr[i], -1, 1))) for i in range(len(Ctr))])
    ctr = Ctr.mean(0); rad = np.linalg.norm(Ctr - ctr, axis=1)
    radte = np.linalg.norm(Cte - ctr, axis=1)
    return dict(n_train=len(Ctr), n_test=len(Cte), med_train_spacing=med_sp,
        scene_radius=float(rad.mean()),
        test_d1_abs=summ(d1), test_d1_norm=summ(d1/med_sp), test_dk_norm=summ(dk/med_sp),
        train_loo_d1_norm=summ(tr_d1/med_sp), train_loo_dk_norm=summ(tr_dk/med_sp),
        ratio_test_over_trainLOO_d1=float(np.median(d1)/np.median(tr_d1)),
        frac_inside_hull=float(inside.mean()), hull_out_norm=summ(outd/med_sp),
        ang_nn1=summ(a1), ang_nn4max=summ(akmax), train_ang_nn1=summ(a1t),
        ratio_ang=float(np.median(a1)/max(np.median(a1t),1e-9)),
        radius_train=summ(rad), radius_test=summ(radte),
        # altitude: use 3rd principal-ish -> use raw axis with largest train std diff later
        )

for s in SCENES:
    root = os.path.join(SET2, s)
    all_p, sp = train_poses(root)
    on_disk = set(os.listdir(os.path.join(root, "train/images")))
    te = load_poses_csv(os.path.join(root, "test/test_poses.csv"))
    te_names = set(t["name"] for t in te)
    tr = [p for p in all_p if p["name"] in on_disk]
    extra = [p for p in all_p if p["name"] not in on_disk and p["name"] not in te_names]
    Ctr, Atr = centers(tr), axes(tr)
    Cte, Ate = centers(te), axes(te)
    r = analyse(s, Ctr, Atr, Cte, Ate)
    r["n_extra_unused_poses"] = len(extra)
    # altitude / up axis: find axis best aligned with scene "vertical". Use PCA -> smallest variance dir for orbit
    Xc = Ctr - Ctr.mean(0)
    u, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    r["pca_sv"] = [float(x) for x in sv/np.sqrt(len(Ctr))]
    up = vt[2]  # least-variance direction of the orbit plane == normal
    htr = Xc @ up; hte = (Cte - Ctr.mean(0)) @ up
    r["alt_train"] = summ(htr); r["alt_test"] = summ(hte)
    r["alt_test_outside_train_frac"] = float(((hte < htr.min()) | (hte > htr.max())).mean())
    # in-plane radius
    P = vt[:2]
    rtr = np.linalg.norm(Xc @ P.T, axis=1); rte = np.linalg.norm((Cte-Ctr.mean(0)) @ P.T, axis=1)
    r["inplane_r_train"] = summ(rtr); r["inplane_r_test"] = summ(rte)
    OUT[s] = r
    print(f"=== {s}  ntr={len(tr)} nte={len(te)} extra_poses={len(extra)}  med_sp={r['med_train_spacing']:.4f}")
    print(f"   test  d1/sp : med={r['test_d1_norm']['median']:.2f}  p90={r['test_d1_norm']['p90']:.2f}  max={r['test_d1_norm']['max']:.2f}")
    print(f"   trainLOO d1/sp: med={r['train_loo_d1_norm']['median']:.2f} p90={r['train_loo_d1_norm']['p90']:.2f}")
    print(f"   test  dk4/sp: med={r['test_dk_norm']['median']:.2f}  p90={r['test_dk_norm']['p90']:.2f}")
    print(f"   inside hull={r['frac_inside_hull']*100:.1f}%   ang_nn1 med={r['ang_nn1']['median']:.2f}deg p90={r['ang_nn1']['p90']:.2f}  (trainLOO {r['train_ang_nn1']['median']:.2f})")
    print(f"   alt train [{r['alt_train']['min']:.2f},{r['alt_train']['max']:.2f}] test [{r['alt_test']['min']:.2f},{r['alt_test']['max']:.2f}] outside={r['alt_test_outside_train_frac']*100:.0f}%")

json.dump(OUT, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_geom_set2_proper.json","w"), indent=1)

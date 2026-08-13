import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *

np.set_printoptions(suppress=True, precision=4)
SCENES = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674", "chair", "bonsai"]

results = {}
for s in SCENES:
    root = os.path.join(SET2, s)
    tr, sp = train_poses(root)
    te = load_poses_csv(os.path.join(root, "test/test_poses.csv"))
    Ctr, Cte = centers(tr), centers(te)
    Atr, Ate = axes(tr), axes(te)
    sp_tr = train_spacing(Ctr)
    med_sp = float(np.median(sp_tr))
    d1, dk, D = nn_stats(Ctr, Cte, 4)
    # train self-holdout baseline: leave-one-out distance of each train cam to other train cams
    Dtt = np.linalg.norm(Ctr[:, None, :] - Ctr[None, :, :], axis=2)
    np.fill_diagonal(Dtt, np.inf)
    Dtts = np.sort(Dtt, axis=1)
    tr_d1, tr_dk = Dtts[:, 0], Dtts[:, :4].mean(axis=1)
    inside, outd = hull_test(Ctr, Cte)
    a1, akmax = angle_to_nn(Ctr, Atr, Cte, Ate, 4)
    # train self angles
    a1t, aktm = angle_to_nn(Ctr, Atr, Ctr, Atr, 5)  # includes self at idx0 -> use k=5 max
    # exclude self: recompute properly
    idx = np.argsort(Dtt, axis=1)
    a1t = np.array([np.degrees(np.arccos(np.clip(Atr[idx[i,0]] @ Atr[i], -1, 1))) for i in range(len(Ctr))])

    scale = float(np.linalg.norm(Ctr - Ctr.mean(0), axis=1).mean())  # rms-ish radius
    extent = Ctr.max(0) - Ctr.min(0)

    results[s] = dict(
        n_train=len(tr), n_test=len(te),
        med_train_spacing=med_sp,
        scene_scale_meanradius=scale,
        extent=[float(x) for x in extent],
        test_d1_abs=summ(d1), test_dk_abs=summ(dk),
        test_d1_norm=summ(d1 / med_sp), test_dk_norm=summ(dk / med_sp),
        train_loo_d1_norm=summ(tr_d1 / med_sp), train_loo_dk_norm=summ(tr_dk / med_sp),
        frac_inside_hull=float(inside.mean()),
        hull_out_dist_norm=summ(outd / med_sp),
        ang_nn1=summ(a1), ang_nn4max=summ(akmax),
        train_ang_nn1=summ(a1t),
        test_names=[t["name"] for t in te],
        train_names=[t["name"] for t in tr],
    )
    print(f"=== {s}: ntr={len(tr)} nte={len(te)} med_sp={med_sp:.4f} scale={scale:.3f}")
    print(f"   test d1/sp  median={np.median(d1/med_sp):.2f} p90={pct(d1/med_sp,90):.2f} max={ (d1/med_sp).max():.2f}")
    print(f"   train LOO d1/sp median={np.median(tr_d1/med_sp):.2f} p90={pct(tr_d1/med_sp,90):.2f}")
    print(f"   inside hull={inside.mean()*100:.1f}%  ang_nn1 med={np.median(a1):.2f} p90={pct(a1,90):.2f}")

json.dump(results, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_geom_set2.json", "w"), indent=1)
print("saved")

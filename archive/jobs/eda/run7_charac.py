import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
from colmap_io import read_cameras_binary, read_points3D_binary, read_images_binary

SET1='/mnt/d/avv/data/phase1/private_set1'
TARGETS = [("set2", os.path.join(SET2, s)) for s in
           ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]] + \
          [("pub", os.path.join(PUB, s)) for s in sorted(os.listdir(PUB)) if os.path.isdir(os.path.join(PUB,s))] + [("set1", os.path.join(SET1, s)) for s in sorted(os.listdir(SET1)) if os.path.isdir(os.path.join(SET1,s))]

OUT = {}
for tag, src in TARGETS:
    s = os.path.basename(src)
    sp = find_sparse(src)
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    ids, xyz, rgb, err, tl = read_points3D_binary(os.path.join(sp, "points3D.bin"))
    all_p, _ = train_poses(src)
    on_disk = sorted(os.listdir(os.path.join(src, "train/images")))
    tr = [p for p in all_p if p["name"] in set(on_disk)]
    Ctr = centers(tr)
    c0 = list(cams.values())[0]
    # sharpness + luminance
    vols, lums, means = [], [], []
    step = max(1, len(on_disk)//80)
    sample = on_disk[::step]
    for n in sample:
        im = cv2.imread(os.path.join(src, "train/images", n), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        if max(im.shape) > 1400:
            f = 1400/max(im.shape); im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        vols.append(float(cv2.Laplacian(im, cv2.CV_64F).var()))
        lums.append(float(im.mean()))
    vols = np.array(vols); lums = np.array(lums)
    # SfM points inside train camera bbox / point density
    ext = Ctr.max(0)-Ctr.min(0)
    rad = np.linalg.norm(Ctr-Ctr.mean(0), axis=1)
    # visible-points per train image
    npv = np.array([p["n3d"] for p in tr]); nkp = np.array([p["nkp"] for p in tr])
    OUT[s] = dict(set=tag, n_train=len(tr), n_sparse=len(all_p),
        W=c0["width"], H=c0["height"], model=c0["model"], params=[float(x) for x in c0["params"]],
        n_cams_models=len(cams),
        n_points=len(ids), mean_track=float(tl.mean()), med_track=float(np.median(tl)),
        mean_reproj_err=float(err.mean()), med_reproj_err=float(np.median(err)),
        pts_per_train_img=float(len(ids)/len(tr)),
        obs_per_img_med=float(np.median(npv)), kp_per_img_med=float(np.median(nkp)),
        registration_rate=float(np.median(npv/np.maximum(nkp,1))),
        scene_radius=float(rad.mean()), extent=[float(x) for x in ext],
        vol_med=float(np.median(vols)), vol_p5=float(np.percentile(vols,5)),
        vol_p95=float(np.percentile(vols,95)), vol_ratio_med_p5=float(np.median(vols)/max(np.percentile(vols,5),1e-9)),
        lum_med=float(np.median(lums)), lum_std=float(lums.std()),
        lum_range=float(lums.max()-lums.min()))
    p = OUT[s]
    print(f"{s:9s}[{tag}] n={p['n_train']:3d} {p['W']}x{p['H']} {p['model']:14s} f={p['params'][0]:.1f} k1={p['params'][-1] if p['model'].startswith('SIMPLE_RADIAL') else float('nan'):+.5f} "
          f"| pts={p['n_points']:7d} trk={p['mean_track']:.1f} err={p['mean_reproj_err']:.3f} pts/img={p['pts_per_train_img']:.0f} "
          f"| VoL med={p['vol_med']:.0f} p5={p['vol_p5']:.0f} | lum {p['lum_med']:.0f}±{p['lum_std']:.1f} rng={p['lum_range']:.0f}")

json.dump(OUT, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_charac.json","w"), indent=1)

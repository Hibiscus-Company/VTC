"""The shipped train/sparse model registers the TEST views with full 2D-3D correspondences.
Project points3D into each test camera using (a) the test_poses.csv intrinsics (pinhole) and
(b) the cameras.bin SIMPLE_RADIAL intrinsics, and compare to the shipped 2D keypoints.
The residual field IS the render-vs-photo misregistration, measured directly at test poses."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
from colmap_io import (read_images_binary, read_points3D_binary, read_cameras_binary, qvec2rotmat)

SCENES=["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]
for s in SCENES:
    src=os.path.join(SET2,s); sp=find_sparse(src)
    cams=read_cameras_binary(os.path.join(sp,"cameras.bin"))
    imgs=read_images_binary(os.path.join(sp,"images.bin"))
    ids,xyz,rgb,err,tl=read_points3D_binary(os.path.join(sp,"points3D.bin"))
    P={int(i):x for i,x in zip(ids,xyz)}
    byname={v["name"]:v for v in imgs.values()}
    te=load_poses_csv(os.path.join(src,"test/test_poses.csv"))
    c0=list(cams.values())[0]
    print(f"=== {s}  cameras.bin: {c0['model']} params={np.round(c0['params'],5)} W={c0['width']} H={c0['height']}  ncams={len(cams)}")
    t0=te[0]
    print(f"    test_poses.csv[0]: fx={t0['fx']:.4f} fy={t0['fy']:.4f} cx={t0['cx']} cy={t0['cy']} w={t0['w']} h={t0['h']}")
    # unique csv intrinsics
    uf=set((round(t['fx'],4),round(t['cx'],2),round(t['cy'],2)) for t in te)
    print(f"    unique csv intrinsics across 60 test poses: {len(uf)}")
    resid_pin, resid_rad = [], []
    for t in te[:20]:
        v=byname[t["name"]]
        R=qvec2rotmat(t["q"]); tv=t["t"]
        pid=v["point3D_ids"]; xys=v["xys"]
        m=pid>=0
        pid=pid[m]; obs=xys[m]
        X=np.array([P[int(p)] for p in pid if int(p) in P])
        keep=np.array([int(p) in P for p in pid])
        obs=obs[keep]
        Xc=(R@X.T).T+tv
        z=Xc[:,2]; ok=z>1e-6
        Xc=Xc[ok]; obs=obs[ok]; z=z[ok]
        xn=Xc[:,0]/z; yn=Xc[:,1]/z
        # (a) pinhole with CSV intrinsics
        up=t["fx"]*xn+t["cx"]; vp=t["fy"]*yn+t["cy"]
        resid_pin.append(np.stack([up-obs[:,0], vp-obs[:,1]],1))
        # (b) SIMPLE_RADIAL with cameras.bin
        pr=c0["params"]
        if c0["model"]=="SIMPLE_RADIAL":
            f,cx,cy,k1=pr
            r2=xn*xn+yn*yn; d=1+k1*r2
            ur=f*xn*d+cx; vr=f*yn*d+cy
        else:
            f,cx,cy=pr[0],pr[1],pr[2]
            ur=f*xn+cx; vr=f*yn+cy
        resid_rad.append(np.stack([ur-obs[:,0], vr-obs[:,1]],1))
    rp=np.concatenate(resid_pin); rr=np.concatenate(resid_rad)
    np_=np.linalg.norm(rp,axis=1); nr=np.linalg.norm(rr,axis=1)
    print(f"    reproj resid vs shipped keypoints (n={len(rp)}):")
    print(f"      CSV-pinhole      : |d| med={np.median(np_):7.3f} p90={np.percentile(np_,90):7.3f}  mean dx={rp[:,0].mean():+.3f} dy={rp[:,1].mean():+.3f}")
    print(f"      cameras.bin model: |d| med={np.median(nr):7.3f} p90={np.percentile(nr,90):7.3f}  mean dx={rr[:,0].mean():+.3f} dy={rr[:,1].mean():+.3f}")

"""Fit the image-coordinate convention (scale+offset) between shipped SfM keypoints and the
CSV/cameras.bin projection, then map the RESIDUAL FIELD over the image plane.
Done for TRAIN views (calibration) and TEST views (the thing we must correct)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
from colmap_io import (read_images_binary, read_points3D_binary, read_cameras_binary, qvec2rotmat)

SCENES=["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]
RES={}
for s in SCENES:
    src=os.path.join(SET2,s); sp=find_sparse(src)
    cams=read_cameras_binary(os.path.join(sp,"cameras.bin"))
    imgs=read_images_binary(os.path.join(sp,"images.bin"))
    ids,xyz,rgb,err,tl=read_points3D_binary(os.path.join(sp,"points3D.bin"))
    P={int(i):x for i,x in zip(ids,xyz)}
    byname={v["name"]:v for v in imgs.values()}
    c0=list(cams.values())[0]; pr=c0["params"]
    te=load_poses_csv(os.path.join(src,"test/test_poses.csv"))
    trn=sorted(os.listdir(os.path.join(src,"train/images")))

    def project(v, use_csv=None):
        q = use_csv["q"] if use_csv else v["qvec"]; tv = use_csv["t"] if use_csv else v["tvec"]
        R=qvec2rotmat(q)
        pid=v["point3D_ids"]; xys=v["xys"]; m=pid>=0
        pid=pid[m]; obs=xys[m]
        keep=np.array([int(p) in P for p in pid])
        pid=pid[keep]; obs=obs[keep]
        X=np.array([P[int(p)] for p in pid])
        Xc=(R@X.T).T+tv; z=Xc[:,2]; ok=z>1e-6
        Xc=Xc[ok]; obs=obs[ok]; z=z[ok]
        xn=Xc[:,0]/z; yn=Xc[:,1]/z
        if c0["model"]=="SIMPLE_RADIAL":
            f,cx,cy,k1=pr; r2=xn*xn+yn*yn; d=1+k1*r2
            u=f*xn*d+cx; vv=f*yn*d+cy
        else:
            f,cx,cy=pr; u=f*xn+cx; vv=f*yn+cy
        return np.stack([u,vv],1), obs

    # fit scale s and offset (ox,oy):  obs ≈ s*pred + o   (least squares over many train views)
    Pp,Oo=[],[]
    for n in trn[:40]:
        p,o=project(byname[n]); Pp.append(p); Oo.append(o)
    Pp=np.concatenate(Pp); Oo=np.concatenate(Oo)
    A=np.stack([Pp[:,0],np.ones(len(Pp))],1); sx,ox=np.linalg.lstsq(A,Oo[:,0],rcond=None)[0]
    A=np.stack([Pp[:,1],np.ones(len(Pp))],1); sy,oy=np.linalg.lstsq(A,Oo[:,1],rcond=None)[0]
    print(f"=== {s}: fitted obs = s*pred + o  ->  sx={sx:.5f} ox={ox:+.4f} | sy={sy:.5f} oy={oy:+.4f}")

    def resid_stats(names, use_csv_map=None, label=""):
        R_=[]; U_=[]
        for n in names:
            v=byname.get(n)
            if v is None: continue
            p,o=project(v, use_csv_map.get(n) if use_csv_map else None)
            # bring obs into shipped-image pixel coords
            oc=np.stack([(o[:,0]-ox)/sx, (o[:,1]-oy)/sy],1)
            R_.append(p-oc); U_.append(p)
        R_=np.concatenate(R_); U_=np.concatenate(U_)
        n=np.linalg.norm(R_,axis=1)
        return R_,U_,n

    csvmap={t["name"]:t for t in te}
    Rtr,Utr,ntr=resid_stats(trn[:60])
    Rte,Ute,nte=resid_stats([t["name"] for t in te], csvmap)
    # radial profile of residual magnitude (is there an uncorrected distortion term?)
    ctr=np.array([c0["width"]/2, c0["height"]/2])
    def radprof(R_,U_):
        r=np.linalg.norm(U_-ctr,axis=1)
        radial=((U_-ctr)*R_).sum(1)/np.maximum(r,1e-6)   # signed radial component
        bins=np.linspace(0,r.max(),6); out=[]
        for i in range(5):
            m=(r>=bins[i])&(r<bins[i+1])
            out.append((float(bins[i]),float(bins[i+1]),int(m.sum()),float(np.median(radial[m])) if m.sum() else np.nan))
        return out
    print(f"    TRAIN resid |d| med={np.median(ntr):.3f}px p90={np.percentile(ntr,90):.3f}  mean(dx,dy)=({Rtr[:,0].mean():+.4f},{Rtr[:,1].mean():+.4f})")
    print(f"    TEST  resid |d| med={np.median(nte):.3f}px p90={np.percentile(nte,90):.3f}  mean(dx,dy)=({Rte[:,0].mean():+.4f},{Rte[:,1].mean():+.4f})")
    print(f"    TEST radial-component median by radius bin (px):")
    for a,b,cnt,md in radprof(Rte,Ute):
        print(f"       r=[{a:6.1f},{b:6.1f}) n={cnt:6d} radial_med={md:+.4f}")
    RES[s]=dict(sx=float(sx),sy=float(sy),ox=float(ox),oy=float(oy),
        train_resid_med=float(np.median(ntr)), test_resid_med=float(np.median(nte)),
        test_bias=[float(Rte[:,0].mean()),float(Rte[:,1].mean())],
        train_bias=[float(Rtr[:,0].mean()),float(Rtr[:,1].mean())])
json.dump(RES, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_field.json","w"), indent=1)

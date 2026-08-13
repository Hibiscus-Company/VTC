import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
from colmap_io import (read_images_binary, read_points3D_binary, read_cameras_binary, qvec2rotmat)

SCENES=["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]
OUT={}
for s in SCENES:
    src=os.path.join(SET2,s); sp=find_sparse(src)
    cams=read_cameras_binary(os.path.join(sp,"cameras.bin")); c0=list(cams.values())[0]; pr=c0["params"]
    imgs=read_images_binary(os.path.join(sp,"images.bin"))
    ids,xyz,rgb,err,tl=read_points3D_binary(os.path.join(sp,"points3D.bin"))
    P={int(i):x for i,x in zip(ids,xyz)}
    byname={v["name"]:v for v in imgs.values()}
    te=load_poses_csv(os.path.join(src,"test/test_poses.csv")); ten=[t["name"] for t in te]
    trn=sorted(os.listdir(os.path.join(src,"train/images")))
    S = 4.0 if c0["width"]==1320 else (1.5 if c0["width"]==720 else 1.0)

    # ---- k2 fit on TRAIN views only (legal), evaluated on TEST views ----
    def gather(names, csvmap=None, lim=60):
        XN,YN,OB=[],[],[]
        for n in names[:lim]:
            v=byname.get(n)
            if v is None: continue
            c=csvmap.get(n) if csvmap else None
            q=c["q"] if c else v["qvec"]; tv=c["t"] if c else v["tvec"]
            R=qvec2rotmat(q)
            pid=v["point3D_ids"]; xys=v["xys"]; m=pid>=0
            pid=pid[m]; obs=xys[m]
            keep=np.array([int(p) in P for p in pid])
            pid=pid[keep]; obs=obs[keep]
            X=np.array([P[int(p)] for p in pid])
            Xc=(R@X.T).T+tv; z=Xc[:,2]; ok=z>1e-6
            Xc=Xc[ok]; obs=obs[ok]/S; z=z[ok]
            XN.append(Xc[:,0]/z); YN.append(Xc[:,1]/z); OB.append(obs)
        return np.concatenate(XN),np.concatenate(YN),np.concatenate(OB)

    def resid(xn,yn,ob,k1,k2):
        r2=xn*xn+yn*yn; d=1+k1*r2+k2*r2*r2
        f,cx,cy=pr[0],pr[1],pr[2]
        return np.stack([f*xn*d+cx-ob[:,0], f*yn*d+cy-ob[:,1]],1)

    k1 = pr[3] if c0["model"]=="SIMPLE_RADIAL" else 0.0
    xn,yn,ob = gather(trn)
    # least-squares refine (k1,k2) on TRAIN
    from scipy.optimize import least_squares
    def fun(p): return resid(xn,yn,ob,p[0],p[1]).ravel()
    sol=least_squares(fun,[k1,0.0],method='lm')
    k1n,k2n=sol.x
    base=np.linalg.norm(resid(xn,yn,ob,k1,0.0),axis=1)
    new =np.linalg.norm(resid(xn,yn,ob,k1n,k2n),axis=1)
    xnT,ynT,obT=gather(ten,{t["name"]:t for t in te},lim=60)
    baseT=np.linalg.norm(resid(xnT,ynT,obT,k1,0.0),axis=1)
    newT =np.linalg.norm(resid(xnT,ynT,obT,k1n,k2n),axis=1)
    print(f"=== {s}  shipped k1={k1:+.6f} -> refit k1={k1n:+.6f} k2={k2n:+.6f}")
    print(f"    TRAIN resid med {np.median(base):.4f} -> {np.median(new):.4f} px  ({100*(1-np.median(new)/np.median(base)):+.1f}%)")
    print(f"    TEST  resid med {np.median(baseT):.4f} -> {np.median(newT):.4f} px  ({100*(1-np.median(newT)/np.median(baseT)):+.1f}%)  [k1,k2 fit on TRAIN only]")

    # ---- blur proxy: keypoints per image, train vs test ----
    kp_tr=np.array([len(byname[n]["point3D_ids"]) for n in trn])
    kp_te=np.array([len(byname[n]["point3D_ids"]) for n in ten])
    obs_tr=np.array([(byname[n]["point3D_ids"]>=0).sum() for n in trn])
    obs_te=np.array([(byname[n]["point3D_ids"]>=0).sum() for n in ten])
    # validate proxy against VoL on train
    step=max(1,len(trn)//60); samp=trn[::step]
    vols=[]; kps=[]
    for n in samp:
        im=cv2.imread(os.path.join(src,"train/images",n),cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        vols.append(cv2.Laplacian(im,cv2.CV_64F).var()); kps.append(len(byname[n]["point3D_ids"]))
    r=np.corrcoef(vols,kps)[0,1]
    print(f"    keypoints/img  TRAIN med={np.median(kp_tr):.0f} p5={np.percentile(kp_tr,5):.0f} | TEST med={np.median(kp_te):.0f} p5={np.percentile(kp_te,5):.0f}  ratio={np.median(kp_te)/np.median(kp_tr):.3f}")
    print(f"    triangulated/img TRAIN med={np.median(obs_tr):.0f} | TEST med={np.median(obs_te):.0f} ratio={np.median(obs_te)/np.median(obs_tr):.3f}")
    print(f"    corr(VoL, keypoints) on train = {r:+.3f}  (proxy validity)")
    OUT[s]=dict(k1=float(k1),k1n=float(k1n),k2n=float(k2n),
        train_resid_before=float(np.median(base)),train_resid_after=float(np.median(new)),
        test_resid_before=float(np.median(baseT)),test_resid_after=float(np.median(newT)),
        kp_train_med=float(np.median(kp_tr)),kp_test_med=float(np.median(kp_te)),
        kp_ratio=float(np.median(kp_te)/np.median(kp_tr)),
        corr_vol_kp=float(r))
json.dump(OUT,open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_k2blur.json","w"),indent=1)

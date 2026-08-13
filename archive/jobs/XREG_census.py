import os,sys,torch,numpy as np,json
M=[("tower_HCM0421_ut42","/mnt/d/avv/r2r9/models/HCM0421_ut42/ckpt.pt",240,1320,989),
   ("tower_HCM0674_ut42","/mnt/d/avv/r2r9/models/HCM0674_ut42/ckpt.pt",240,1320,989),
   ("tower_HCM0674_r28","/mnt/d/avv/r28_members/HCM0674/ckpt.pt",240,1320,989),
   ("chair_ut42","/mnt/d/avv/r2r9/models/chair_ut42/ckpt.pt",205,720,1280),
   ("bonsai_K1_noUT_aa","/mnt/d/avv/bonsai_eval/K1_noUT_aa/ckpt.pt",248,1920,1080),
   ("bonsai_K4_pC_seed7","/mnt/d/avv/bonsai_eval/K4_pC_seed7/ckpt.pt",248,1920,1080)]
out=[]
for name,p,ni,W,H in M:
    if not os.path.exists(p): print("MISSING",p,flush=True); continue
    c=torch.load(p,map_location="cpu",weights_only=False)
    sp=c["splats"] if "splats" in c else c
    S=torch.exp(sp["scales"]).double(); o=torch.sigmoid(sp["opacities"].double()); mu=sp["means"].double()
    tot=S.shape[0]; m=(o>0.05)
    Sx=S[m].numpy(); Ss=np.sort(Sx,axis=1)[:,::-1]
    # scene extent: robust radius = median distance to median center, and 5-95 pct bbox diag
    ctr=mu.median(0).values; d=(mu-ctr).norm(dim=1).numpy()
    r50=float(np.percentile(d,50)); r95=float(np.percentile(d,95))
    ms=float(S.mean())                 # THE reg term: loss += scale_reg * exp(scales).mean()
    r=dict(name=name,total=int(tot),alive=int(m.sum()),alivefrac=float(m.double().mean()),
      gpx=float(tot)/(ni*W*H), alive_gpx=float(m.sum())/(ni*W*H),
      s2s3=float(np.median(Ss[:,1]/np.maximum(Ss[:,2],1e-30))),
      s2s1=float(np.median(Ss[:,1]/Ss[:,0])),
      s1s3=float(np.median(Ss[:,0]/np.maximum(Ss[:,2],1e-30))),
      medop=float(o.median()),meanop=float(o.mean()),
      mean_exp_scale=ms, regterm=0.01*ms, r50=r50,r95=r95,
      mean_scale_over_r95=ms/r95,
      med_maxscale=float(np.median(Ss[:,0])), med_minscale=float(np.median(Ss[:,2])))
    out.append(r); print(json.dumps(r),flush=True)
    del c,sp,S,o,mu,Sx,Ss
json.dump(out,open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/XREG_census.json","w"),indent=1)

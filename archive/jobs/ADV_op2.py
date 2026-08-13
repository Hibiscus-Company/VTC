import torch, os, json
P=[("capD_5M_rs15k_ns8k","/mnt/d/avv/bonsai_sel/capD_5Mearly/ckpt.pt"),
   ("r33_8M_rs15k_ns8k","/mnt/d/avv/r33_bonsai/eval_s42/ckpt.pt")]
for name,p in P:
    ck=torch.load(p,map_location="cpu",weights_only=False)
    d=ck["splats"] if isinstance(ck,dict) and "splats" in ck else ck
    o=torch.sigmoid(d["opacities"].float().flatten()); sc=torch.exp(d["scales"].float())
    N=o.numel(); mass=o.sum().item()
    so,_=torch.sort(o,descending=True); cs=torch.cumsum(so,0)
    mx=sc.max(dim=1).values
    print(json.dumps(dict(name=name,N=N,dead_le005=float((o<=0.005).float().mean()),
      gt05=int((o>0.05).sum()),gt1=int((o>0.1).sum()),gt5=int((o>0.5).sum()),
      median=float(o.median()),mean=float(o.mean()),mass=mass,
      n90pct_mass=int((cs<0.90*mass).sum())+1,
      med_maxscale=float(mx.median()),mass_x_area=float((o*mx.pow(2)).sum()))),flush=True)
    del ck,d,o,sc,mx

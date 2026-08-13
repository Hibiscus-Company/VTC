import torch, os, json
P = [
 ("capD_5M_30k_rs15k_ns8k",  "/mnt/d/avv/bonsai_sel/capD_5Mearly/ckpt.pt"),
 ("r33_8M_30k_rs15k_ns8k",   "/mnt/d/avv/r33_bonsai/eval_s42/ckpt.pt"),
 ("SHIPPED_pc_ut42_5M",      "/mnt/d/avv/bonsai_pc/ut42/ckpt.pt"),
]
for name, p in P:
    if not os.path.exists(p):
        print(json.dumps(dict(name=name, err="MISSING")), flush=True); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    d = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck
    if not (isinstance(d, dict) and "opacities" in d):
        for k in list(d.keys()):
            if isinstance(d[k], dict) and "opacities" in d[k]: d = d[k]; break
    o = torch.sigmoid(d["opacities"].float().flatten())
    sc = torch.exp(d["scales"].float())
    N = o.numel(); mass = o.sum().item()
    so,_ = torch.sort(o, descending=True); cs = torch.cumsum(so,0)
    n90 = int((cs < .90*mass).sum())+1
    mx = sc.max(dim=1).values
    print(json.dumps(dict(name=name, N=N,
        dead_le005=float((o<=0.005).float().mean()),
        gt05=int((o>0.05).sum()), gt1=int((o>0.1).sum()), gt5=int((o>0.5).sum()),
        median=float(o.median()), mean=float(o.mean()), mass=mass, n90=n90,
        med_maxscale=float(mx.median()), mass_x_area=float((o*mx.pow(2)).sum()))), flush=True)
    del ck,d,o,sc,so,cs,mx
print("ADVDONE", flush=True)

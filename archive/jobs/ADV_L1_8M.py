import torch, json
p="/mnt/d/avv/r33_bonsai/eval_s42/ckpt.pt"
c=torch.load(p, map_location="cpu", weights_only=False)
sp=c.get("splats",c)
o=sp["opacities"] if "opacities" in sp else sp["opacity"]
o=o.float().flatten()
if o.min()<0: o=torch.sigmoid(o)
sc=sp["scales"].float()
if sc.max()<0 or sc.min()<-5: sc=torch.exp(sc)
ms=sc.max(dim=1).values
r=dict(name="ADV_8M", N=int(o.numel()),
  dead_le005=float((o<=0.005).float().mean()),
  lt05=float((o<0.05).float().mean()),
  gt05=int((o>0.05).sum()), gt1=int((o>0.1).sum()), gt5=int((o>0.5).sum()),
  median=float(o.median()), mean=float(o.mean()), mass=float(o.sum()),
  med_maxscale=float(ms.median()))
print(json.dumps(r), flush=True)

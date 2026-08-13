import torch
d=torch.load('/mnt/d/avv/bonsai_final/ut42/ckpt.pt',map_location='cpu',weights_only=False)
sp=d['splats'] if 'splats' in d else d
o=torch.sigmoid(sp['opacities'].float().flatten()); N=o.numel()
print("N=%d  median=%.4e  mean=%.4e"%(N,o.median(),o.mean()))
for t in [0.005,0.015,0.05]:
    print("  frac o<=%-6g : %.4f"%(t,(o<=t).float().mean()))
band=((o>0.005)&(o<=0.015))
print("BAND (0.005,0.015]  -> the population a 3.0x downward shift pushes across the dead line")
print("  count %d  frac %.4f"%(band.sum(),band.float().mean()))
print("  its share of total alpha-mass: %.4f"%(o[band].sum()/o.sum()))
print("  alive (o>0.005) count %d  frac %.4f"%((o>0.005).sum(),(o>0.005).float().mean()))
# what a uniform 3x opacity shrink does to the dead fraction
print("AFTER uniform o/3 (sim-measured equilibrium shift for reg-dominated pop):")
print("  frac o<=0.005 : %.4f  (delta +%.4f)"%(((o/3)<=0.005).float().mean(),((o/3)<=0.005).float().mean()-(o<=0.005).float().mean()))
print("  alpha-mass retained: %.4f"%((o/3)[(o/3)>0.005].sum()/o[o>0.005].sum()))

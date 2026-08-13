import torch, sys, math
p = sys.argv[1]
ck = torch.load(p, map_location='cpu', weights_only=False)
sp = ck['splats'] if 'splats' in ck else ck
if not isinstance(sp, dict):
    print(type(ck), list(ck.keys())[:10]); sys.exit()
print("keys:", list(sp.keys()))
op = torch.sigmoid(sp['opacities'].float().flatten())
sc = torch.exp(sp['scales'].float())
N = op.numel()
gate = torch.sigmoid(-100.0*(op-0.005))
print(f"N={N}")
for th in [0.5,0.1,0.01]:
    print(f"  frac(gate>{th}) = {(gate>th).float().mean().item():.4f}")
print(f"  mean gate = {gate.mean().item():.4f}")
print(f"  frac(op<0.005) = {(op<0.005).float().mean().item():.4f}   median op = {op.median().item():.4f}")
# opacity-mass fraction subject to noise
print(f"  opacity-mass gated frac = {(gate*op).sum().item()/op.sum().item():.6f}")
# area-weighted (projected footprint ~ prod of 2 largest scales) * opacity
s,_ = torch.sort(sc, dim=1, descending=True)
area = s[:,0]*s[:,1]
w = area*op
print(f"  alpha*area-weighted gated frac = {(gate*w).sum().item()/w.sum().item():.6f}")
# displacement magnitude: |disp| ~ smax^2 * gate * noise_scale ; ratio to splat size = smax*gate*noise_scale
for step,total in [(30000,60000),(40000,60000),(50000,60000)]:
    ns = 1.6e-4*(0.01**(step/total))*5e5
    ratio = s[:,0]*gate*ns
    print(f"  step {step}: noise_scale={ns:.3f}  frac(|disp|>0.5*own size)={ (ratio>0.5).float().mean().item():.5f}  frac(>0.1)={(ratio>0.1).float().mean().item():.5f}")
print(f"  scale stats: median smax={s[:,0].median().item():.5f}  p99={s[:,0].kthvalue(int(0.99*N))[0].item():.5f}")

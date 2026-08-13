import os, sys, time
import numpy as np, torch
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
torch.set_num_threads(20)
from lapfuse import lap_pyr, _K

# ---- 1. white-noise gain of the finest Laplacian band
g = torch.Generator().manual_seed(0)
x = torch.randn(1,3,512,512, generator=g)
laps,res,sizes = lap_pyr(x, 5, _K)
print("g0 (L0 white-noise gain) =", float((laps[0]**2).mean()/ (x**2).mean()))
for i,l in enumerate(laps):
    print("  level",i,"gain", float((l**2).mean()/(x**2).mean()), "shape", tuple(l.shape))

# ---- 2. nan reproduction in mip3d_apply (float32)
def mip3d_apply(scales_lin, opac, filt):
    s2 = scales_lin.square()
    d2 = s2 + filt.square()
    coef = torch.sqrt(s2.prod(-1) / d2.prod(-1).clamp_min(1e-30))
    return torch.sqrt(d2), opac * coef

def stable(scales_lin, opac, filt):
    s2 = scales_lin.square(); d2 = s2 + filt.square()
    coef = torch.exp(0.5*(torch.log(s2.clamp_min(1e-45)) - torch.log(d2)).sum(-1))
    return torch.sqrt(d2), opac*coef

filt = torch.tensor([[0.00448407]])
tests = {}
for name, s in [("normal 1e-3", 1e-3), ("tiny 1e-5", 1e-5), ("tiny 1e-7", 1e-7),
                ("tiny 3e-8", 3e-8), ("tiny 1e-8", 1e-8), ("big 1e6", 1e6),
                ("big 2.7e6", 2.7e6), ("big 1e7", 1e7), ("big 1e13", 1e13),
                ("huge 2e19", 2e19), ("huge 1e20", 1e20)]:
    sc = torch.full((1,3), float(s))
    _, o = mip3d_apply(sc, torch.tensor([0.5]), filt)
    _, o2 = stable(sc, torch.tensor([0.5]), filt)
    print(f"{name:>12} scale={s:9.3g}  shipped opac={o.item():.6g}   stable={o2.item():.6g}")
# anisotropic: 2 big 1 small
for a,b,c in [(1e7,1e7,1e-4),(1e12,1e12,1e12),(1e13,1e2,1e2),(1e14,1.0,1.0),(1e20,1e-3,1e-3)]:
    sc = torch.tensor([[a,b,c]])
    _,o = mip3d_apply(sc, torch.tensor([0.5]), filt)
    _,o2 = stable(sc, torch.tensor([0.5]), filt)
    print(f"aniso {a:9.3g},{b:9.3g},{c:9.3g} shipped={o.item():.6g} stable={o2.item():.6g}")

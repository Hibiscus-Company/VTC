import numpy as np
# Exact replication of the opacity path: torch.optim.Adam(lr=5e-2, eps=1e-15, betas=(0.9,0.999))
# param = logit-opacity. reg grad = opacity_reg * sigmoid'(z)/N = c*o*(1-o)/N   (loss uses .mean())
# data grad: sparse+stochastic, N(0,s) with prob p (gaussian visible/contributing), else 0.
N=5_000_000; LR=5e-2; EPS=1e-15; B1,B2=0.9,0.999; STEPS=30000
def run(c, s, p, z0, seed):
    rng=np.random.default_rng(seed); z=z0; m=0.0; v=0.0
    for t in range(1,STEPS+1):
        o=1/(1+np.exp(-z))
        g = c*o*(1-o)/N
        if rng.random()<p: g += rng.normal(0,s)
        m=B1*m+(1-B1)*g; v=B2*v+(1-B2)*g*g
        mh=m/(1-B1**t); vh=v/(1-B2**t)
        z-= LR*mh/(np.sqrt(vh)+EPS)
        if z<-30: z=-30.0
    return z,1/(1+np.exp(-z))
print("z0=logit(0.5)=0  (fresh/relocated gaussian). 30000 steps.")
print(f"{'data-grad s':>12} {'p_vis':>6} | {'z_end c=0.01':>13} {'o_end':>10} | {'z_end c=0.03':>13} {'o_end':>10} | {'ratio dz':>9}")
for s,p in [(0.0,0.0),(1e-13,0.05),(1e-12,0.05),(1e-11,0.05),(1e-10,0.05),(1e-9,0.05),(1e-8,0.05),(1e-7,0.05)]:
    z1,o1=run(0.01,s,p,0.0,1); z3,o3=run(0.03,s,p,0.0,1)
    r = (z3/z1) if abs(z1)>1e-12 else float('nan')
    print(f"{s:12.1e} {p:6.2f} | {z1:13.4f} {o1:10.3e} | {z3:13.4f} {o3:10.3e} | {r:9.3f}")

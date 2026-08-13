import torch, numpy as np, time, sys
t0=time.time()
p = "/mnt/d/avv/r2r9/models/HCM0539_ut77/ckpt.pt"
try:
    d = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
except Exception as e:
    print("mmap failed:", e); d = torch.load(p, map_location="cpu", weights_only=False)
sp = d["splats"]
print("keys", list(sp.keys()), "load", time.time()-t0)
s = sp["scales"]            # log scales
o = sp["opacities"]
print("N", s.shape[0])
sl = torch.exp(s.float())
print("log-scale  min %.4f  max %.4f  p99.999 %.4f" % (float(s.min()), float(s.max()),
      float(torch.quantile(s.flatten()[::97].float(), 0.99999))))
print("lin-scale  min %.4g  max %.4g" % (float(sl.min()), float(sl.max())))
# product of the three linear scales, in float64 for the diagnosis
prod = sl.double().prod(-1)
print("prod(s) max %.6g   #>1.845e19 (float32 s2.prod overflow) = %d" % (float(prod.max()), int((prod>1.845e19).sum())))
print("#>1e10 %d  #>1e6 %d  #>1e3 %d  #>10 %d" % (int((prod>1e10).sum()), int((prod>1e6).sum()),
      int((prod>1e3).sum()), int((prod>10).sum())))
print("#any nan in scales %d, opac %d, means %d" % (int(torch.isnan(s).sum()),
      int(torch.isnan(o).sum()), int(torch.isnan(sp["means"]).sum())))
print("#inf scales %d" % int(torch.isinf(sl).sum()))
# underflow side
print("#prod(s)<3.7e-23 (s2.prod underflow to 0) = %d" % int((prod<3.7e-23).sum()))
print("#prod(s^2) subnormal (<1.18e-38 i.e. prod(s)<1.09e-19) = %d  (%.4f%%)" %
      (int((prod<1.09e-19).sum()), 100*float((prod<1.09e-19).float().mean())))
q=[0.0,.001,.01,.5,.99,.999,.9999,1.0]
f=sl.flatten()[::13].double()
print("lin scale quantiles", [float(torch.quantile(f, x)) for x in q])

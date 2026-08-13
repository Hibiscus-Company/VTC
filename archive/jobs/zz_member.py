import os, sys, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
N=8
def ld(p): return np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.
print(f"{'scene':>9} {'member':>10} {'PSNRvsPeers':>12} {'p99.99|e|':>10} {'%|e|>0.3':>9} {'%|e|>0.5':>9} {'mean':>7}")
for T in ["HCM0421","HCM0539","HCM0540"]:
    peers = [f"/mnt/d/avv/r25_mip3d/{T}/test_png",
             f"/mnt/d/avv/r2r9/models/{T}_ut42/test_png",
             f"/mnt/d/avv/r22_seed101/{T}/test_png"]
    cand  = f"/mnt/d/avv/r28_members/{T}/test_png"
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(cand) if f.endswith(".png"))[:N]
    for name, D, P in [("r28new", cand, peers),
                       ("mip3d", peers[0], [cand]+peers[1:]),
                       ("ut42",  peers[1], [cand,peers[0],peers[2]])]:
        se=0.; tail=[]; big3=0; big5=0; tot=0; mu=0.
        for s in stems:
            a=ld(os.path.join(D,s+".png"))
            pm=np.mean([ld(os.path.join(d,s+".png")) for d in P],0)
            e=np.abs(a-pm); se+=float(((a-pm)**2).mean()); tail.append(np.quantile(e,0.9999))
            big3+=int((e>0.3).sum()); big5+=int((e>0.5).sum()); tot+=e.size; mu+=a.mean()
        n=len(stems)
        print(f"{T:>9} {name:>10} {10*np.log10(1/(se/n)):12.3f} {np.mean(tail):10.4f} "
              f"{100*big3/tot:9.5f} {100*big5/tot:9.5f} {mu/n:7.4f}")

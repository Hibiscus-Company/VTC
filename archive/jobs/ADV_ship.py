import os,sys,numpy as np,tempfile,shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
B="/mnt/d/avv/bonsai_eval"
POOL=["K1_noUT_aa","K4_pC_seed1k","K4_pC_seed7","eps10","eps20","ppisp_pc"]
D={a:f"{B}/{a}/eval_png" for a in POOL}
D["sr001seed42"]="/mnt/d/avv/r36_shape/sr001seed42/eval_png"
D["sr0.03"]="/mnt/d/avv/r35_scalereg/sr0.03/eval_png"
for k,v in D.items():
    n=len([f for f in os.listdir(v) if f.endswith('.png')]) if os.path.isdir(v) else -1
    print(k,v,n,flush=True)
SETS={"P6":POOL,"P7c":POOL+["sr001seed42"],"P7t":POOL+["sr0.03"]}
out=[]
for tag,members in SETS.items():
    td=tempfile.mkdtemp(prefix=f"ADV_{tag}_",dir="/mnt/d/avv")
    st=sorted(f[:-4] for f in os.listdir(D[members[0]]) if f.endswith(".png"))
    for s in st:
        a=np.mean([np.asarray(Image.open(f"{D[m]}/{s}.png").convert("RGB"),dtype=np.float64) for m in members],0)
        Image.fromarray(np.clip(a+0.5,0,255).astype(np.uint8)).save(f"{td}/{s}.png")
    out.append(f"{td}={tag}")
    print("built",tag,td,len(st),flush=True)
print("ARGS "+" ".join(out),flush=True)
open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV_ship_args.txt","w").write(" ".join(out))

"""GT-free magnitude of the r33 change (2) on the REAL scene: encode the shipped r33 HCM0421
masters at q99 and q100 and measure distortion vs the master + reproduction of the shipped zips."""
import io,os,zipfile,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
MD="/mnt/d/avv/r33/tower_ens/HCM0421/png"
ARMS={"q99ss2":dict(quality=99,subsampling=2,optimize=True,progressive=True),
      "q100ss2":dict(quality=100,subsampling=2,optimize=True,progressive=True)}
z33=zipfile.ZipFile('/mnt/d/avv/submissions/sub_round33_deadband.zip')
z32=zipfile.ZipFile('/mnt/d/avv/submissions/sub_round32_videolam.zip')
n33={i.filename.split('/')[-1]:i for i in z33.infolist() if i.filename.startswith("HCM0421/")}
n32={i.filename.split('/')[-1]:i for i in z32.infolist() if i.filename.startswith("HCM0421/")}
stems=sorted(f for f in os.listdir(MD) if f.endswith(".png"))
mse={k:[] for k in ARMS}; byt={k:0 for k in ARMS}; cross=[]; repro100=0; repro99_vs_r32=0; n=0
d3332=[]
for f in stems[:20]:
    u8=np.asarray(Image.open(os.path.join(MD,f)).convert("RGB"))
    dec={}
    for k,kw in ARMS.items():
        b=io.BytesIO(); Image.fromarray(u8).save(b,"JPEG",**kw); raw=b.getvalue(); byt[k]+=len(raw)
        j=np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.float64)
        dec[k]=j; mse[k].append(((j-u8.astype(np.float64))**2).mean())
        if k=="q100ss2":
            jn=f.replace(".png",".JPG")
            if jn in n33 and raw==z33.read("HCM0421/"+jn): repro100+=1
    cross.append(((dec["q99ss2"]-dec["q100ss2"])**2).mean())
    # shipped r33 (q100, deadband masters) vs shipped r32 (q99, old masters)
    jn=f.replace(".png",".JPG")
    if jn in n33 and jn in n32:
        a=np.asarray(Image.open(io.BytesIO(z33.read("HCM0421/"+jn))).convert("RGB")).astype(np.float64)
        b_=np.asarray(Image.open(io.BytesIO(z32.read("HCM0421/"+jn))).convert("RGB")).astype(np.float64)
        d3332.append(((a-b_)**2).mean())
    n+=1
for k in ARMS: print(f"HCM0421 {k:8s} selfRMSE_vs_master {np.sqrt(np.mean(mse[k])):.4f} LSB   MiB/img {byt[k]/n/2**20:.3f}")
print(f"HCM0421 q99-vs-q100 decode RMSE {np.sqrt(np.mean(cross)):.4f} LSB   n={n}")
print(f"shipped r33 reproduced from masters at q100: {repro100}/{n} byte-exact")
print(f"shipped r33 vs shipped r32 total RMSE (deadband+qchange): {np.sqrt(np.mean(d3332)):.4f} LSB")
print(f"byte delta q100 vs q99 on these masters: {100*(byt['q100ss2']-byt['q99ss2'])/byt['q99ss2']:+.2f}%")

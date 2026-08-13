"""Write a sparse/0 whose images.bin contains ONLY the images physically present in images/.
The delivered model lists all 276 bonsai frames (248 train + 28 held-out); the 3DGS-lineage
loader tries to open every one. Filtering the model is the correct fix -- copying the held-out
images in would be training on the eval holes."""
import os,struct,shutil,sys
src,dst=sys.argv[1],sys.argv[2]
imgdir=os.path.join(os.path.dirname(os.path.dirname(src)),"images")
have={f for f in os.listdir(imgdir)}
os.makedirs(dst,exist_ok=True)
for f in ("cameras.bin","points3D.bin","points3D.ply"):
    p=os.path.join(src,f)
    if os.path.exists(p): shutil.copy2(p,os.path.join(dst,f))
recs=[]
with open(os.path.join(src,"images.bin"),"rb") as f:
    n=struct.unpack("<Q",f.read(8))[0]
    for _ in range(n):
        head=f.read(64)
        nm=b""
        while True:
            c=f.read(1)
            if c==b"\x00": break
            nm+=c
        k=struct.unpack("<Q",f.read(8))[0]
        pts=f.read(24*k)
        if nm.decode() in have: recs.append((head,nm,k,pts))
with open(os.path.join(dst,"images.bin"),"wb") as f:
    f.write(struct.pack("<Q",len(recs)))
    for head,nm,k,pts in recs:
        f.write(head); f.write(nm); f.write(b"\x00"); f.write(struct.pack("<Q",k)); f.write(pts)
print(f"filtered {n} -> {len(recs)} images (images/ has {len(have)})")

#!/bin/bash
set -e
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
M=/mnt/d/avv/r33_bonsai/prod_s555
PRI=/mnt/d/avv/data/phase1/private_set2
conda activate fastgs2
echo "=== opacity gate (weights_only=False) ==="
python - "$M/ckpt.pt" <<'PY'
import sys, torch
c=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
sp=c.get("splats",c); o=sp["opacities"] if "opacities" in sp else sp["opacity"]
o=o.float().flatten(); o=torch.sigmoid(o) if o.min()<0 else o
print(f"GATE n={o.numel()} median_opacity={float(o.median()):.4f} dead_frac={float((o<0.005).float().mean()):.4f}")
sys.exit(0 if (float(o.median())>0.02 and float((o<0.005).float().mean())<0.60) else 1)
PY
conda activate gsplat
echo "=== render 28 TEST views ==="
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $PRI/bonsai/test/test_poses.csv --out $M/test_render --png_dir $M/test_png 2>&1 | tail -2
echo "=== render TRAIN views (legal-GT quality check vs the old 5M members) ==="
python - <<'PY'
import csv,os
src="/mnt/d/avv/data/phase1/private_set2/bonsai/train/images"
import struct
def qs(p):
    out=[]
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            d=struct.unpack('<idddddddi',f.read(64)); nm=b''
            while True:
                ch=f.read(1)
                if ch==b'\x00': break
                nm+=ch
            k=struct.unpack('<Q',f.read(8))[0]; f.read(24*k)
            out.append((nm.decode(),d))
    return out
rows=qs("/mnt/d/avv/data/phase1/private_set2/bonsai/train/sparse/0/images.bin")
have=set(os.listdir(src))
sel=[r for r in rows if r[0] in have][::16][:16]
import csv as C
w=C.writer(open("/mnt/d/avv/r33_bonsai/train_sub.csv","w",newline=""))
w.writerow(["image_name","qw","qx","qy","qz","tx","ty","tz","fx","fy","cx","cy","width","height"])
for nm,d in sel:
    w.writerow([nm,d[1],d[2],d[3],d[4],d[5],d[6],d[7],1108.5124,1108.5124,960.0,540.0,1920,1080])
print("wrote",len(sel),"train poses")
PY
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv /mnt/d/avv/r33_bonsai/train_sub.csv --out $M/trainchk_render --png_dir $M/trainchk_png 2>&1 | tail -2
echo "=== DONE ==="; ls $M/test_png/*.png | wc -l; ls $M/trainchk_png/*.png 2>/dev/null | wc -l
touch /mnt/d/avv/render_bonsai8m.DONE

import os, sys, struct, numpy as np
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 read_points3D_binary, qvec2rotmat)
from scipy.spatial import cKDTree
from PIL import Image

def load_scene(sp):
    imgs = read_extrinsics_binary(os.path.join(sp, "images.bin"))
    cam = list(read_intrinsics_binary(os.path.join(sp, "cameras.bin")).values())[0]
    xyz, rgb, errs = read_points3D_binary(os.path.join(sp, "points3D.bin"))
    return imgs, cam, xyz, errs

def intr(cam):
    if cam.model == "SIMPLE_PINHOLE": f, cx, cy = cam.params; return f, f, cx, cy
    if cam.model == "PINHOLE": fx, fy, cx, cy = cam.params; return fx, fy, cx, cy
    return cam.params[0], cam.params[0], cam.params[1], cam.params[2]

def track_lengths(path):
    tl = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8); f.read(24); f.read(3); f.read(8)
            m = struct.unpack("<Q", f.read(8))[0]; f.read(8*m); tl.append(m)
    return np.array(tl)

BON = "/mnt/d/avv/data/phase1/private_set2/bonsai/train/sparse/0"
imgs, cam, xyz, errs = load_scene(BON)
tl = track_lengths(os.path.join(BON, "points3D.bin"))
print("BONSAI  N=%d  reproj err mean %.3f med %.3f  track mean %.2f med %d  >=3 %.0f%%  ==2 %.0f%%"
      % (len(xyz), errs.mean(), np.median(errs), tl.mean(), np.median(tl), 100*(tl>=3).mean(), 100*(tl==2).mean()))
fx, fy, cx, cy = intr(cam); W, H = cam.width, cam.height

GY, GX = 8, 12
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
ARM = "/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_render"
byname = {im.name: im for im in imgs.values()}
nnacc = np.zeros((GY, GX)); eacc = np.zeros((GY, GX)); gacc = np.zeros((GY, GX)); n = 0
per = []
for fn in sorted(os.listdir(GT)):
    st = os.path.splitext(fn)[0]
    im = byname.get(fn) or byname.get(st + ".jpg")
    rp = os.path.join(ARM, st + ".jpg")
    if im is None or not os.path.exists(rp): print("skip", st); continue
    R = qvec2rotmat(im.qvec); t = im.tvec
    P = xyz @ R.T + t; z = P[:, 2]; m = z > 1e-6
    u = fx*P[m,0]/z[m]+cx; v = fy*P[m,1]/z[m]+cy
    k = (u>=-50)&(u<W+50)&(v>=-50)&(v<H+50)
    tree = cKDTree(np.stack([u[k], v[k]], 1))
    S = 8
    gy = np.linspace(0, H, GY*S, endpoint=False)+H/(GY*S*2)
    gx = np.linspace(0, W, GX*S, endpoint=False)+W/(GX*S*2)
    QY, QX = np.meshgrid(gy, gx, indexing="ij")
    d, _ = tree.query(np.stack([QX.ravel(), QY.ravel()], 1))
    dcell = d.reshape(GY, S, GX, S).mean(axis=(1,3))
    g = np.asarray(Image.open(os.path.join(GT, fn)).convert("RGB"), np.float32)
    r = np.asarray(Image.open(rp).convert("RGB"), np.float32)
    e = np.abs(g-r).mean(2)
    ys = np.linspace(0,H,GY+1).astype(int); xs = np.linspace(0,W,GX+1).astype(int)
    ec = np.array([[e[ys[a]:ys[a+1], xs[b]:xs[b+1]].mean() for b in range(GX)] for a in range(GY)])
    # GT high-frequency energy per cell (texture proxy)
    gg = g.mean(2); gh = np.abs(np.diff(gg,axis=0,prepend=gg[:1]))+np.abs(np.diff(gg,axis=1,prepend=gg[:,:1]))
    gc = np.array([[gh[ys[a]:ys[a+1], xs[b]:xs[b+1]].mean() for b in range(GX)] for a in range(GY)])
    nnacc += dcell; eacc += ec; gacc += gc; n += 1
    per.append((st, k.sum(), dcell.mean(), e.mean()))
nn = nnacc/n; ee = eacc/n; gg_ = gacc/n
np.set_printoptions(precision=1, suppress=True, linewidth=230)
print("\n=== mean px distance to nearest SfM point (n=%d eval views) ===" % n); print(nn)
print("\n=== mean |render-GT| (0-255) per cell, arm K4_pC_seed7 ==="); np.set_printoptions(precision=2, suppress=True, linewidth=230); print(ee)
print("\n=== GT high-freq energy per cell ==="); print(gg_)
a, b, c = nn.ravel(), ee.ravel(), gg_.ravel()
print("\ncorr(NNdist, err)  = %+.3f" % np.corrcoef(a, b)[0,1])
print("corr(NNdist, GThf) = %+.3f" % np.corrcoef(a, c)[0,1])
print("corr(GThf,   err)  = %+.3f" % np.corrcoef(c, b)[0,1])
# partial corr of NNdist with err controlling for GT high-freq
def pc(x,y,z):
    bx=np.polyfit(z,x,1); by=np.polyfit(z,y,1)
    return np.corrcoef(x-np.polyval(bx,z), y-np.polyval(by,z))[0,1]
print("partial corr(NNdist, err | GThf) = %+.3f" % pc(a,b,c))
pa = np.array([[p[1],p[2],p[3]] for p in per])
print("\nper-image n=%d: corr(ptsInView, err)=%+.3f  corr(meanNN, err)=%+.3f" %
      (len(pa), np.corrcoef(pa[:,0],pa[:,2])[0,1], np.corrcoef(pa[:,1],pa[:,2])[0,1]))

# --- reference scenes ---
for nm, path in [("chair", "/mnt/d/avv/data/phase1/private_set2/chair/train/sparse/0")]:
    try:
        i2, c2, x2, e2 = load_scene(path)
        t2 = track_lengths(os.path.join(path,"points3D.bin"))
        f2x,f2y,c2x,c2y = intr(c2); W2,H2 = c2.width,c2.height
        ds=[]; nv=[]
        for im in sorted(i2.values(), key=lambda i:i.name)[::30]:
            R=qvec2rotmat(im.qvec); t=im.tvec; P=x2@R.T+t; z=P[:,2]; m=z>1e-6
            u=f2x*P[m,0]/z[m]+c2x; v=f2y*P[m,1]/z[m]+c2y
            k=(u>=-50)&(u<W2+50)&(v>=-50)&(v<H2+50)
            if k.sum()<20: continue
            tr=cKDTree(np.stack([u[k],v[k]],1))
            gy=np.linspace(0,H2,64,endpoint=False)+H2/128; gx=np.linspace(0,W2,64,endpoint=False)+W2/128
            QY,QX=np.meshgrid(gy,gx,indexing="ij")
            d,_=tr.query(np.stack([QX.ravel(),QY.ravel()],1)); ds.append(d.mean()); nv.append(k.sum())
        print("\n%-8s N=%d  %dx%d  reprojerr %.3f  track mean %.2f  ptsInView %d  meanNN %.1f px"
              % (nm, len(x2), W2, H2, e2.mean(), t2.mean(), np.mean(nv), np.mean(ds)))
    except Exception as ex:
        print("skip", nm, ex)
tow = "/mnt/d/avv/data/phase1/private_set2"
for nm in sorted(os.listdir(tow)):
    p = os.path.join(tow, nm, "train", "sparse", "0")
    if nm in ("bonsai","chair") or not os.path.isdir(p): continue
    try:
        i2,c2,x2,e2 = load_scene(p); t2 = track_lengths(os.path.join(p,"points3D.bin"))
        f2x,f2y,c2x,c2y = intr(c2); W2,H2=c2.width,c2.height
        ds=[];nv=[]
        for im in sorted(i2.values(), key=lambda i:i.name)[::40]:
            R=qvec2rotmat(im.qvec); t=im.tvec; P=x2@R.T+t; z=P[:,2]; m=z>1e-6
            u=f2x*P[m,0]/z[m]+c2x; v=f2y*P[m,1]/z[m]+c2y
            k=(u>=-50)&(u<W2+50)&(v>=-50)&(v<H2+50)
            if k.sum()<20: continue
            tr=cKDTree(np.stack([u[k],v[k]],1))
            gy=np.linspace(0,H2,64,endpoint=False)+H2/128; gx=np.linspace(0,W2,64,endpoint=False)+W2/128
            QY,QX=np.meshgrid(gy,gx,indexing="ij"); d,_=tr.query(np.stack([QX.ravel(),QY.ravel()],1))
            ds.append(d.mean()); nv.append(k.sum())
        print("%-8s N=%d  %dx%d  reprojerr %.3f  track mean %.2f  ptsInView %d  meanNN %.1f px"
              % (nm, len(x2), W2, H2, e2.mean(), t2.mean(), np.mean(nv), np.mean(ds)))
    except Exception as ex:
        print("skip", nm, ex)

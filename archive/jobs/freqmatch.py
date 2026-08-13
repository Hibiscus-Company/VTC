"""DISCRIMINATOR: are trained Gaussians COARSER or FINER than the texture they must represent?

If COARSER  -> the densification criterion is the bottleneck (SAD-GS-class fix, cheap Python).
If FINER but the render is still blurry -> the kernel/rasteriser is the bottleneck (CUDA work).

Compares, per image cell:
  (a) dominant local texture period of the TRAIN PHOTO (legal GT), from a Laplacian pyramid
      argmax-over-levels, and
  (b) the finest screen-space extent actually available from the trained Gaussians projected
      into that same view (EWA: Sigma2D = J R Sigma R^T J^T).
Scene-vs-scene comparison is convention-independent, which is the load-bearing output.
"""
import os, sys, struct, numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(4)

def qvec2R(q):
    w,x,y,z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]], dtype=np.float64)

def read_images_bin(p):
    out = {}
    with open(p,'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            d = struct.unpack('<idddddddi', f.read(64))
            nm = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                nm += c
            k = struct.unpack('<Q', f.read(8))[0]; f.read(24*k)
            out[nm.decode()] = (np.array(d[1:5]), np.array(d[5:8]))
    return out

def tex_period(img_gray, nlev=6):
    """Dominant texture period per pixel: argmax over Laplacian levels of |L_k|, level->period."""
    import cv2
    g = img_gray.astype(np.float32)
    cur = g; resp = []
    for k in range(nlev):
        blur = cv2.GaussianBlur(cur, (0,0), 1.0)
        lap  = cur - blur
        r = np.abs(lap)
        for _ in range(k):                       # bring back to full res
            r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        resp.append(cv2.resize(np.abs(lap), (g.shape[1], g.shape[0]),
                               interpolation=cv2.INTER_LINEAR))
        cur = cv2.resize(blur, (blur.shape[1]//2, blur.shape[0]//2),
                         interpolation=cv2.INTER_AREA)
        if min(cur.shape) < 8: break
    R = np.stack(resp, 0)                        # [L,H,W]
    rms = R.reshape(len(R),-1).std(1).reshape(-1,1,1) + 1e-8
    R = R / rms                                  # WHITEN: natural images are ~1/f, so raw
                                                 # |Laplacian| always peaks at coarse scales
    lev = R.argmax(0).astype(np.float32)
    return (2.0 ** (lev + 2.0)), R               # period in px: level0 -> 4px, level1 -> 8px ...

def gauss_extent(ckpt_path, qt, cam, W, H, opa_thresh=0.05, chunk=1_000_000):
    """Return (u, v, sigma_minor_px, opacity) for Gaussians visible in this view."""
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sp = c["splats"]
    K = c.get("K", None); wh = c.get("wh", None)
    means = sp["means"]; scales = torch.exp(sp["scales"]); quats = sp["quats"]
    opac  = torch.sigmoid(sp["opacities"])
    keep  = opac > opa_thresh
    means = means[keep].double().numpy(); scales = scales[keep].double().numpy()
    quats = quats[keep].double().numpy();  opac = opac[keep].double().numpy()
    q, t = qt
    Rw = qvec2R(q); Tw = t
    fx, fy, cx, cy = cam
    us=[]; vs=[]; sm=[]; op=[]
    N = len(means)
    for s in range(0, N, chunk):
        e = min(s+chunk, N)
        X = means[s:e]; S = scales[s:e]; Q = quats[s:e]; O = opac[s:e]
        Xc = X @ Rw.T + Tw
        z = Xc[:,2]
        m = z > 0.2
        if not m.any(): continue
        Xc = Xc[m]; S = S[m]; Q = Q[m]; O = O[m]; z = z[m]
        u = fx*Xc[:,0]/z + cx; v = fy*Xc[:,1]/z + cy
        inb = (u > -64) & (u < W+64) & (v > -64) & (v < H+64)
        if not inb.any(): continue
        Xc=Xc[inb]; S=S[inb]; Q=Q[inb]; O=O[inb]; z=z[inb]; u=u[inb]; v=v[inb]
        Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        w_,x_,y_,z_ = Qn[:,0],Qn[:,1],Qn[:,2],Qn[:,3]
        Rg = np.empty((len(Qn),3,3))
        Rg[:,0,0]=1-2*(y_*y_+z_*z_); Rg[:,0,1]=2*(x_*y_-w_*z_); Rg[:,0,2]=2*(x_*z_+w_*y_)
        Rg[:,1,0]=2*(x_*y_+w_*z_);  Rg[:,1,1]=1-2*(x_*x_+z_*z_); Rg[:,1,2]=2*(y_*z_-w_*x_)
        Rg[:,2,0]=2*(x_*z_-w_*y_);  Rg[:,2,1]=2*(y_*z_+w_*x_);  Rg[:,2,2]=1-2*(x_*x_+y_*y_)
        M = Rg * S[:,None,:]                              # R @ diag(s)
        Sig = M @ np.transpose(M,(0,2,1))                 # world 3x3
        Sc = Rw @ Sig @ Rw.T                              # camera 3x3
        zz = Xc[:,2]
        J = np.zeros((len(zz),2,3))
        J[:,0,0]=fx/zz; J[:,0,2]=-fx*Xc[:,0]/zz**2
        J[:,1,1]=fy/zz; J[:,1,2]=-fy*Xc[:,1]/zz**2
        S2 = J @ Sc @ np.transpose(J,(0,2,1))
        a=S2[:,0,0]+0.3; b=S2[:,0,1]; d=S2[:,1,1]+0.3     # gsplat low-pass dilation
        tr=a+d; det=a*d-b*b
        disc=np.sqrt(np.maximum(tr*tr/4-det,0))
        lmin=np.maximum(tr/2-disc,1e-8)
        us.append(u); vs.append(v); sm.append(np.sqrt(lmin)); op.append(O)
    if not us: return None
    return (np.concatenate(us),np.concatenate(vs),np.concatenate(sm),np.concatenate(op))

def run(tag, ckpt, scene, cam, nviews=3, cell=16):
    D=f"/mnt/d/avv/data/phase1/private_set2/{scene}"
    imgs=read_images_bin(f"{D}/train/sparse/0/images.bin")
    have=set(os.listdir(f"{D}/train/images"))
    names=sorted(n for n in imgs if n in have)[::max(1,len(have)//nviews)][:nviews]
    import cv2
    print(f"\n{'='*92}\n{tag}  ckpt={os.path.basename(os.path.dirname(ckpt))}  n={len(names)} views, cell={cell}px")
    allr=[]
    for nm in names:
        im=np.asarray(Image.open(f"{D}/train/images/{nm}").convert("L"),dtype=np.float32)
        H,W=im.shape
        per,_=tex_period(im)
        g=gauss_extent(ckpt, imgs[nm], cam, W, H)
        if g is None: print("  no gaussians"); continue
        u,v,sig,op=g
        ch,cw=H//cell,W//cell
        ui=np.clip((u//cell).astype(int),0,cw-1); vi=np.clip((v//cell).astype(int),0,ch-1)
        idx=vi*cw+ui
        o=np.lexsort((sig,idx)); si=idx[o]; ss=sig[o]
        uniq,start,cnt=np.unique(si,return_index=True,return_counts=True)
        pick=start+(0.10*cnt).astype(int)           # 10th pct, not the extreme min
        fine=np.full(ch*cw,np.inf); fine[uniq]=ss[pick]
        f2=fine.reshape(ch,cw)
        pc=cv2.resize(per,(cw,ch),interpolation=cv2.INTER_AREA)
        ok=np.isfinite(f2)
        if ok.sum()==0:
            print(f'  {nm[:34]:34s} NO CELLS  (proj u range {u.min():.0f}..{u.max():.0f}, v {v.min():.0f}..{v.max():.0f}, n={len(u)})'); continue
        # representable period of a gaussian of std sigma ~ 4*sigma (its -3dB support)
        ratio=(4.0*f2[ok])/np.maximum(pc[ok],1e-6)
        allr.append(ratio)
        print(f"  {nm[:34]:34s} cells {ok.sum():5d}  sigma_min med {np.median(f2[ok]):6.3f}px  "
              f"texperiod med {np.median(pc[ok]):6.2f}px  ratio med {np.median(ratio):6.3f}")
    r=np.concatenate(allr)
    print(f"  --> RATIO (4*sigma_finest / texture_period): med {np.median(r):.3f}  "
          f"p25 {np.percentile(r,25):.3f}  p75 {np.percentile(r,75):.3f}  "
          f"frac>1 (gaussians COARSER than texture) {100*np.mean(r>1):.1f}%")
    return r

CAM_BONSAI=(1108.5124,1108.5124,960.0,540.0)
import csv
row=next(csv.DictReader(open("/mnt/d/avv/data/phase1/private_set2/HCM0421/test/test_poses.csv")))
CAM_TOWER=(float(row["fx"]),float(row["fy"]),float(row["cx"]),float(row["cy"]))
print("tower intrinsics from csv:",CAM_TOWER)
rb=run("BONSAI (shipped capD 5M)","/mnt/d/avv/bonsai_sel/capD_5Mearly/ckpt.pt","bonsai",CAM_BONSAI)
rt=run("TOWER HCM0421 (ut42 8M)","/mnt/d/avv/r2r9/models/HCM0421_ut42/ckpt.pt","HCM0421",CAM_TOWER)
print(f"\n{'='*92}\nVERDICT INPUT: bonsai median ratio {np.median(rb):.3f} vs tower {np.median(rt):.3f}")
print("  ratio > 1  => gaussians coarser than the texture they must carry => densification is the bottleneck")
print("  ratio < 1  => gaussians already finer than the texture => the kernel/rasteriser is")

"""EXP-2  CHROMA-SUBSPACE lever.

HARD FACT (measured): every GT photo, train and test, is a JPEG with sampling
[(1,2,2,0),(2,1,1,1),(3,1,1,1)] = 4:2:0.  Its Cb/Cr planes are physically HALF resolution,
upsampled by the decoder.  Our renders carry FULL-resolution chroma, so every bit of chroma
detail finer than 2px is guaranteed-wrong energy.  Projecting the render's chroma onto the
same subspace the GT lives in can only remove error that cannot possibly be right.

Step A  identify the decoder's chroma upsample kernel by testing which down/up round trip is
        IDEMPOTENT on the GT itself (a true 4:2:0 image is a fixed point of its own chain).
Step B  decompose the render->GT error into luma and chroma to size the prize.
Step C  apply the projection to the production ensemble and score.
"""
import numpy as np, hlib, io, json
from PIL import Image

hlib.init()
N = hlib.names(); R = hlib.renders(); GU8 = np.asarray(hlib.gt())
G = GU8.astype(np.float32) / 255.0
MEM = ['gsplatB9ut', 'gsplatB10ut8M', 'gsplatB11ut60k', 'gsplatB12ut8Ms7']
base = np.zeros(G.shape, np.float32)
for m in MEM:
    base += R[N.index(m)].astype(np.float32) / 255.0
base /= len(MEM)
n, H, W, _ = base.shape

# ---- JPEG (BT.601 full-range) RGB<->YCbCr, float ----
M = np.array([[0.299, 0.587, 0.114],
              [-0.168736, -0.331264, 0.5],
              [0.5, -0.418688, -0.081312]], np.float64)
Minv = np.linalg.inv(M)
OFF = np.array([0.0, 0.5, 0.5])


def to_ycc(x):
    return x.astype(np.float64) @ M.T + OFF


def to_rgb(y):
    return (y - OFF) @ Minv.T


def pad_even(a):
    H, W = a.shape[:2]
    ph, pw = H % 2, W % 2
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)) + ((0, 0),) * (a.ndim - 2), mode='edge')
    return a, H, W


def down_box(c):
    a, H0, W0 = pad_even(c)
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3)), H0, W0


def up_nearest(s, H0, W0):
    return np.repeat(np.repeat(s, 2, 0), 2, 1)[:H0, :W0]


def up_fancy(s, H0, W0):
    """libjpeg 'fancy' (triangular) chroma upsample: out = (3*near + far)/4 per axis."""
    h, w = s.shape
    e = np.pad(s, 1, mode='edge')
    out = np.empty((h * 2, w * 2), s.dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            # neighbour offset: -1 for the first sub-sample, +1 for the second
            oy, ox = (-1 if dy == 0 else 1), (-1 if dx == 0 else 1)
            c00 = e[1:1 + h, 1:1 + w]
            c10 = e[1 + oy:1 + oy + h, 1:1 + w]
            c01 = e[1:1 + h, 1 + ox:1 + ox + w]
            c11 = e[1 + oy:1 + oy + h, 1 + ox:1 + ox + w]
            out[dy::2, dx::2] = (9 * c00 + 3 * c10 + 3 * c01 + c11) / 16.0
    return out[:H0, :W0]


def up_bilinear(s, H0, W0):
    im = Image.fromarray(s.astype(np.float32), mode='F')
    return np.asarray(im.resize((s.shape[1] * 2, s.shape[0] * 2), Image.BILINEAR),
                      dtype=np.float64)[:H0, :W0]


UPS = {'nearest': up_nearest, 'fancy': up_fancy, 'bilinear': up_bilinear}


def project_chroma(img, up='fancy'):
    """Project one HxWx3 RGB float image onto the 4:2:0 chroma subspace."""
    y = to_ycc(img)
    out = y.copy()
    for c in (1, 2):
        s, H0, W0 = down_box(y[:, :, c])
        out[:, :, c] = UPS[up](s, H0, W0)
    return to_rgb(out)


# =================== Step A: which upsample makes GT idempotent? ===================
print("=== Step A: GT chroma round-trip residual (a true 4:2:0 image is a FIXED POINT) ===")
for up in UPS:
    r = []
    for i in range(0, n, 6):
        p = project_chroma(G[i], up)
        r.append(np.sqrt(((p - G[i]) ** 2).mean()) * 255)
    print(f"  GT self round-trip rmse ({up:9s}) = {np.mean(r):.4f}/255")
# control: same op on the RENDER (full-res chroma) should move it a lot more
r = [np.sqrt(((project_chroma(base[i], 'fancy') - base[i]) ** 2).mean()) * 255
     for i in range(0, n, 6)]
print(f"  RENDER round-trip rmse (fancy)      = {np.mean(r):.4f}/255   <- how much it moves us")

# =================== Step B: luma / chroma error decomposition ===================
print("\n=== Step B: where does the error live? ===")
eb = to_ycc(base) - to_ycc(G)
tot = (eb ** 2).mean()
print(f"  MSE share  Y {(eb[...,0]**2).mean()/tot*100:5.1f}%   "
      f"Cb {(eb[...,1]**2).mean()/tot*100:5.1f}%   Cr {(eb[...,2]**2).mean()/tot*100:5.1f}%")
# how much of the chroma error is in the >2px band we are about to delete?
hf = 0.0; lf = 0.0
for i in range(0, n, 6):
    p = project_chroma(base[i], 'fancy')
    yp, yb, yg = to_ycc(p), to_ycc(base[i]), to_ycc(G[i])
    hf += ((yb[..., 1:] - yp[..., 1:]) ** 2).mean()
    lf += ((yp[..., 1:] - yg[..., 1:]) ** 2).mean()
print(f"  chroma err removable by projection {hf/(hf+lf)*100:5.1f}%  (rest survives)")

# =================== Step C: score it ===================
print("\n=== Step C: production-harness score ===")
res = {}
res['base'] = hlib.score(base, GU8, "BASE k4 float-mean")
for up in ['fancy', 'nearest', 'bilinear']:
    proj = np.stack([project_chroma(base[i], up) for i in range(n)]).astype(np.float32)
    res[up] = hlib.score(proj, GU8, f"chroma420 proj ({up})")
    if up == 'fancy':
        np.save('/home/bkai/.claude/jobs/1c9cf7e9/tmp/tj/base_chroma_fancy.npy', proj)

b = res['base']['score']
print("\n%-34s %8s %9s" % ("config", "score", "dScore"))
for k, v in res.items():
    print("%-34s %8.4f  %+8.4f" % (v['tag'], v['score'], v['score'] - b))
json.dump({k: v for k, v in res.items()},
          open('/home/bkai/.claude/jobs/1c9cf7e9/tmp/tj/exp2.json', 'w'), indent=1)

import numpy as np, PIL.Image as I, os, glob
vs = sorted(glob.glob('/mnt/d/avv/output/HCM0181_*/test_poses_renders_png'))
names = [v.split('/')[-2] for v in vs]
fn = 'DJI_20241229103340_0026_V.png'
rng = np.random.RandomState(0)
idx = rng.choice(989*1320*3, 20000, replace=False)
X = []
for v in vs:
    a = np.asarray(I.open(os.path.join(v, fn)), np.float64).ravel()[idx]
    X.append(a)
X = np.stack(X, 1)
y = np.asarray(I.open('/mnt/d/avv/prodharness/k4/png/' + fn), np.float64).ravel()[idx]
w, res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
for n, wi in sorted(zip(names, w), key=lambda t: -abs(t[1])):
    print(f'{wi:8.4f}  {n}')
print('resid rms', np.sqrt(np.mean((X@w-y)**2)))

import os, sys
import numpy as np, cv2
sys.path.insert(0, "/mnt/d/avv/metric_probe")
import mlib

GTD = "/mnt/d/avv/data/phase1/public_set/{s}/test/images"
K4 = "/mnt/d/avv/prodharness/k4/png"
MEM = ["HCM0181_gsplatB9ut", "HCM0181_gsplatB10ut8M",
       "HCM0181_gsplatB11ut60k", "HCM0181_gsplatB12ut8Ms7"]
ROOT = "/mnt/d/avv/output"


def apply_field(img8, field, scale=1.0):
    H, W, _ = img8.shape
    fu = cv2.resize(field * scale, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(img8.astype(np.float32) / 255.0,
                    (xx + fu[..., 0]).astype(np.float32),
                    (yy + fu[..., 1]).astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def gt_map(scene):
    d = GTD.format(s=scene)
    return d, {os.path.splitext(f)[0]: f for f in sorted(os.listdir(d))}


def stems(scene="HCM0181"):
    return [os.path.splitext(f)[0] for f in sorted(os.listdir(K4))]

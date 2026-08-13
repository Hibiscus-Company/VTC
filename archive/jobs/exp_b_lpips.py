"""(b) LPIPS-vgg per-scale decomposition of our deficit on the production harness."""
import sys, json
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from harness import *

LAYERS = ["relu1_2 (64ch, 1x)", "relu2_2 (128ch, 1/2)", "relu3_3 (256ch, 1/4)",
          "relu4_3 (512ch, 1/8)", "relu5_3 (512ch, 1/16)"]

SETS = [(f"{s}/B9ut", R_TEST.format(s=s), GT_TEST.format(s=s)) for s in SCENES]
SETS.append(("HCM0181/k4", K4, GT_TEST.format(s="HCM0181")))

out = {}
print(f"{'set':16s} " + " ".join(f"{l.split()[0]:>9s}" for l in LAYERS) + f" {'TOTAL':>9s}")
for tag, rd, gd in SETS:
    acc = np.zeros(5); n = 0
    with torch.no_grad():
        for stem, rp, gp in pairs(rd, gd):
            r = load(rp).to(DEV); g = load(gp).to(DEV)
            acc += lpips_layers(r, g); n += 1
    acc /= n
    out[tag] = acc.tolist()
    print(f"{tag:16s} " + " ".join(f"{v:9.5f}" for v in acc) + f" {acc.sum():9.5f}")
    print(f"{'  %of total':16s} " + " ".join(f"{100*v/acc.sum():8.1f}%" for v in acc))
    print(f"{'  score pts':16s} " + " ".join(f"{40*v:9.4f}" for v in acc) + f" {40*acc.sum():9.4f}")

# controls: what does each scale respond to?  degrade the GT in known ways.
print("\n# CONTROLS on HCM0181 GT (what each scale is sensitive to)")
import torch.nn.functional as Fn
gd = GT_TEST.format(s="HCM0181")
gps = [p for _, _, p in pairs(K4, gd)][:20]


def gauss1d(sig, ks=None):
    ks = ks or int(2 * round(3 * sig) + 1)
    x = torch.arange(ks, dtype=torch.float32) - ks // 2
    k = torch.exp(-x ** 2 / (2 * sig ** 2)); k /= k.sum()
    return k.to(DEV), ks


def blur(x, sig):
    k, ks = gauss1d(sig)
    k2 = k.view(1, 1, 1, ks).expand(3, 1, 1, ks)
    x = Fn.conv2d(Fn.pad(x, (ks // 2,) * 2 + (0, 0), mode="reflect"), k2, groups=3)
    k2 = k.view(1, 1, ks, 1).expand(3, 1, ks, 1)
    return Fn.conv2d(Fn.pad(x, (0, 0) + (ks // 2,) * 2, mode="reflect"), k2, groups=3)


ctrls = {
    "blur sigma=0.5": lambda g: blur(g, 0.5),
    "blur sigma=1.0": lambda g: blur(g, 1.0),
    "blur sigma=2.0": lambda g: blur(g, 2.0),
    "gauss noise .01": lambda g: (g + 0.01 * torch.randn_like(g)).clamp(0, 1),
    "gauss noise .03": lambda g: (g + 0.03 * torch.randn_like(g)).clamp(0, 1),
    "shift 0.5px": lambda g: Fn.grid_sample(
        g, (torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, g.shape[-2], device=DEV),
            torch.linspace(-1, 1, g.shape[-1], device=DEV), indexing="ij")[::-1], -1)[None]
            + torch.tensor([1.0 / g.shape[-1], 0.0], device=DEV)),
        mode="bilinear", padding_mode="border", align_corners=True),
    "gain +2%": lambda g: (g * 1.02).clamp(0, 1),
}
torch.manual_seed(0)
for name, fn in ctrls.items():
    acc = np.zeros(5); n = 0
    with torch.no_grad():
        for gp in gps:
            g = load(gp).to(DEV); r = fn(g)
            acc += lpips_layers(r, g); n += 1
    acc /= n
    print(f"{name:16s} " + " ".join(f"{v:9.5f}" for v in acc) + f" {acc.sum():9.5f}"
          + "   share " + " ".join(f"{100*v/max(acc.sum(),1e-9):4.0f}%" for v in acc))
    out["ctrl:" + name] = acc.tolist()

json.dump(out, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/lpips_layers.json", "w"), indent=1)

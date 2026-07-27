#!/usr/bin/env python
"""Difix3D+ zero-shot pass over a render dir (H-A test). Pads to /8, runs the released
single-step pipeline (nvidia/difix), unpads, saves PNG. --strength blends output with input
(1.0 = full Difix, 0.3 = conservative) since our renders are far cleaner than the
heavy-artifact regime Difix was trained on."""
import argparse, os
import numpy as np
from PIL import Image
import torch

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--prompt", default="remove degradation")
    args = ap.parse_args()
    from pipeline_difix import DifixPipeline  # from the Difix3D repo (pip-installed dir or cwd)

    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(args.in_dir))
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for i, f in enumerate(files):
        img = Image.open(os.path.join(args.in_dir, f)).convert("RGB")
        W, H = img.size
        W8, H8 = (W + 7) // 8 * 8, (H + 7) // 8 * 8
        if (W8, H8) != (W, H):
            pad = Image.new("RGB", (W8, H8))
            pad.paste(img, (0, 0))
            src = pad
        else:
            src = img
        with torch.no_grad():
            out = pipe(args.prompt, image=src, num_inference_steps=1,
                       timesteps=[199], guidance_scale=0.0).images[0]
        out = out.crop((0, 0, W, H))
        if args.strength < 1.0:
            a = np.asarray(img, dtype=np.float32)
            b = np.asarray(out, dtype=np.float32)
            out = Image.fromarray(
                np.clip((1 - args.strength) * a + args.strength * b + 0.5, 0, 255
                        ).astype(np.uint8))
        out.save(os.path.join(args.out_dir, os.path.splitext(f)[0] + ".png"))
        if i % 10 == 0:
            print(f"[{i}/{len(files)}] {f}")
    print(f"difix pass done: {len(files)} images -> {args.out_dir} (strength {args.strength})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Verify a prepared VAR 2026 environment. Exit 0 = ready."""
import sys, glob, os, subprocess
fail = []
try:
    import torch
except Exception as e:
    print("FATAL: torch missing:", e); sys.exit(1)
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    fail.append("no CUDA device visible")
else:
    cap = torch.cuda.get_device_capability(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} | {gb:.1f} GiB")
    if gb < 15.0: fail.append(f"only {gb:.1f} GiB VRAM; the pipeline peaks near 13.5 GiB and needs >= 16 GB")
    try:
        import gsplat
        print("gsplat", getattr(gsplat, "__version__", "?"))
        so = glob.glob(os.path.dirname(gsplat.__file__) + "/**/csrc*.so", recursive=True)
        if not so:
            fail.append("gsplat kernels NOT built yet -- first use will JIT-compile; nvcc must be present")
        else:
            out = subprocess.run(["cuobjdump", "--list-elf", so[0]], capture_output=True, text=True).stdout
            tgt = f"sm_{cap[0]}{cap[1]}"
            print(f"gsplat kernels: {so[0].split('/')[-1]}, covers {tgt}: {'YES' if tgt in out else 'NO'}")
            if tgt not in out: fail.append(f"gsplat kernels do not cover {tgt} -- rebuild gsplat")
    except Exception as e:
        fail.append(f"gsplat import failed: {e}")
if subprocess.run(["which", "nvcc"], capture_output=True).returncode != 0:
    fail.append("nvcc NOT on PATH -- gsplat JIT-compiles at runtime and will fail")
else:
    print("nvcc:", subprocess.run(["nvcc","--version"],capture_output=True,text=True).stdout.strip().splitlines()[-1])
for m in ("lpips","fused_ssim","numpy","PIL","plyfile","cv2","scipy","tqdm"):
    try: __import__(m)
    except Exception as e: fail.append(f"missing module {m}: {e}")
print()
if fail:
    print("NOT READY:"); [print("  -", f) for f in fail]; sys.exit(1)
print("ENVIRONMENT READY"); sys.exit(0)

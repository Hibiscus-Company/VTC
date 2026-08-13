"""A1 grid: baseline + unsharp grid + radial spectral match, per frame, FULL RES, CPU only.
One JSON per (frame,setting) in CACHE -> resumable, and extra settings can be appended later."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp

REN = "/mnt/d/avv/r36_shape/sr01/eval_png"
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
CACHE = "/mnt/d/avv/r42_bonsai78/a1_oracle/cache"

SIGMAS = [0.6, 1.0, 1.6, 2.5]
ALPHAS = [0.15, 0.3, 0.5, 0.8, 1.2]


def base_settings():
    S = ["base"]
    for s in SIGMAS:
        for a in ALPHAS:
            S.append(f"us_{s}_{a}")
    S += ["spec_pc", "spec_sh"]
    return S


def frames():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(REN) if f.endswith(".png"))


def worker(args):
    stems, settings_for = args
    import cpu_metrics as M
    n = 0
    for stem in stems:
        keys = [k for k in settings_for[stem]
                if not os.path.exists(os.path.join(CACHE, f"{stem}__{k}.json"))]
        if not keys:
            continue
        r0 = M.load(os.path.join(REN, stem + ".png"))
        g0 = M.load(os.path.join(GT, stem + ".jpg"))
        assert r0.shape == g0.shape, (stem, r0.shape, g0.shape)
        fg = M.gt_feats(g0)
        for key in keys:
            rec = {"stem": stem, "key": key}
            if key == "base":
                r = r0.clamp(0, 1)
                rec["lapvar_render"] = M.lapvar(r0)
                rec["lapvar_gt"] = M.lapvar(g0)
            elif key.startswith("us_"):
                _, s, a = key.split("_")
                r = M.unsharp(r0, float(s), float(a))
            elif key.startswith("u8_"):          # uint8 round-trip of an unsharp setting
                _, s, a = key.split("_")
                r = M.quantize8(M.unsharp(r0, float(s), float(a)))
            elif key == "spec_pc":
                r, gains = M.radial_gain_apply(r0, g0, per_channel=True)
                rec["gain_curve_g"] = [round(float(v), 5) for v in gains[1][::8].tolist()]
            elif key == "spec_sh":
                r, gains = M.radial_gain_apply(r0, g0, per_channel=False)
                rec["gain_curve"] = [round(float(v), 5) for v in gains[0][::8].tolist()]
            elif key == "specw_pc":
                r, gains = M.radial_wiener_apply(r0, g0, per_channel=True)
                rec["gain_curve_g"] = [round(float(v), 5) for v in gains[1][::8].tolist()]
            elif key == "specw_sh":
                r, gains = M.radial_wiener_apply(r0, g0, per_channel=False)
                rec["gain_curve"] = [round(float(v), 5) for v in gains[0][::8].tolist()]
            elif key == "spec_pc_u8":
                r, _ = M.radial_gain_apply(r0, g0, per_channel=True)
                r = M.quantize8(r)
            else:
                raise ValueError(key)
            t = time.monotonic()
            p, s_, l = M.metrics_fast(r, g0, fg)
            rec.update(psnr=p, ssim=s_, lpips=l, secs=round(time.monotonic() - t, 2))
            out = os.path.join(CACHE, f"{stem}__{key}.json")
            with open(out + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(out + ".tmp", out)
            n += 1
        del fg
    return n


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    extra = [x for x in (sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []) if x]
    os.makedirs(CACHE, exist_ok=True)
    F = frames()
    S = base_settings() + extra
    settings_for = {f: S for f in F}
    pending = sum(1 for f in F for k in S
                  if not os.path.exists(os.path.join(CACHE, f"{f}__{k}.json")))
    print(f"{pending} jobs pending ({len(F)} frames x {len(S)} settings)", flush=True)
    if not pending:
        return
    chunks = [(F[i::nw], settings_for) for i in range(nw)]
    t0 = time.monotonic()
    with mp.Pool(nw) as pool:
        res = pool.map(worker, chunks)
    print("done", sum(res), "in", round(time.monotonic() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

import json, numpy as np
d=json.load(open('/home/bkai/.claude/jobs/1c9cf7e9/tmp/eda/out_charac.json'))
# Known measured values on the 5 PUBLIC scenes (EXPERIMENTS.md line ~1200, gsplatB9ut, recipe fixed)
pub_psnr={"HCM0181":24.27,"HCM0193":25.68,"HCM0204":24.89,"hcm0031":25.09,"hcm0034":25.14}
pub_train={"HCM0181":26.04,"HCM0193":27.56,"HCM0204":27.13,"hcm0031":27.37,"hcm0034":27.14}
pub_score={"HCM0181":73.98,"HCM0193":73.61,"HCM0204":74.19,"hcm0031":73.09,"hcm0034":74.53}  # exp00

feats=["vol_med","vol_p5","pts_per_train_img","n_points","mean_track","mean_reproj_err","obs_per_img_med"]
names=list(pub_psnr)
print("Feature correlations with measured PUBLIC per-scene results (n=5):")
print(f"{'feature':22s} {'r vs TEST psnr':>15s} {'r vs TRAIN psnr':>16s} {'r vs score':>11s}")
for f in feats:
    x=np.array([d[n][f] for n in names])
    print(f"{f:22s} {np.corrcoef(x,[pub_psnr[n] for n in names])[0,1]:15.3f} "
          f"{np.corrcoef(x,[pub_train[n] for n in names])[0,1]:16.3f} "
          f"{np.corrcoef(x,[pub_score[n] for n in names])[0,1]:11.3f}")

# fit TEST psnr ~ a*vol_med + b on the 5 public towers, extrapolate to set2 towers
x=np.array([d[n]["vol_med"] for n in names]); y=np.array([pub_psnr[n] for n in names])
a,b=np.polyfit(x,y,1); r=np.corrcoef(x,y)[0,1]
print(f"\nfit: TEST_PSNR = {a:.6f}*VoL_med + {b:.3f}   (r={r:.3f}, n=5 -- WEAK, treat as indicative)")
print(f"{'scene':10s} {'VoL_med':>8s} {'pred TEST psnr':>15s} {'delta vs HCM0181 pred':>22s}")
base=a*d["HCM0181"]["vol_med"]+b
for s in ["HCM0181","HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"]:
    p=a*d[s]["vol_med"]+b
    print(f"{s:10s} {d[s]['vol_med']:8.0f} {p:15.2f} {p-base:+22.2f}")
print(f"\nmean predicted set2-tower PSNR advantage over HCM0181: "
      f"{np.mean([a*d[s]['vol_med']+b for s in ['HCM0421','HCM0539','HCM0540','HCM0644','HCM0674']])-base:+.2f} dB")
print(f"  -> score contribution via 0.3*PSNR/50*100 term only: "
      f"{0.3*(np.mean([a*d[s]['vol_med']+b for s in ['HCM0421','HCM0539','HCM0540','HCM0644','HCM0674']])-base)/50*100:+.2f} pts/scene")

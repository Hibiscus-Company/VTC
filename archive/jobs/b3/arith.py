#!/usr/bin/env python
"""B3 Job-1: the arithmetic to bonsai=78 on the eval holes. Pure arithmetic, no data."""
import math

def score(P, S, L):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))

# --- the two baselines in circulation -------------------------------------------------
K1 = dict(P=26.9814, S=0.8503, L=0.2448)          # parent-brief header (r19 K1 arm)
SR01 = dict(P=26.9464, S=0.84780, L=0.24027)      # r36_shape/sr01, BEST single arm ever
INS = dict(P=27.941, S=0.8707, L=0.2047)          # bonsai TRAIN-view (in-sample) fit, n=16

print("=== baselines ===")
for n, d in (("K1 (brief header)", K1), ("sr01 (best single)", SR01), ("in-sample train views", INS)):
    print(f"  {n:24s} P {d['P']:7.4f}  S {d['S']:.4f}  L {d['L']:.4f}  -> {score(**d):.4f}")

BASE = K1
b = score(**BASE)
TGT = 78.0
need = TGT - b
print(f"\nBASE = {b:.4f}   TARGET = {TGT}   NEED = +{need:.4f}")
print(f"(from sr01 {score(**SR01):.4f} the need is +{TGT-score(**SR01):.4f})")

# --- single-axis routes ---------------------------------------------------------------
print("\n=== ROUTES TO 78 (each holds the other two metrics at baseline) ===")
dL = -need / 40.0
Lt = BASE['L'] + dL
print(f"LPIPS-only : dLPIPS {dL:+.5f} -> LPIPS {Lt:.5f}   ({100*(1-Lt/BASE['L']):.1f}% reduction; "
      f"vs in-sample {INS['L']:.4f} = {INS['L']/Lt:.2f}x better than the model's own train fit)")
dP = need / 0.6
Pt = BASE['P'] + dP
print(f"PSNR-only  : dPSNR {dP:+.4f} dB -> PSNR {Pt:.4f} dB  (MSE x1/{10**(dP/10):.2f}; "
      f"vs in-sample {INS['P']:.3f} dB = +{Pt-INS['P']:.2f} dB above the train fit)")
dS = need / 30.0
St = BASE['S'] + dS
print(f"SSIM-only  : dSSIM {dS:+.5f} -> SSIM {St:.5f}  ** IMPOSSIBLE, SSIM<=1 **")
print(f"             SSIM=1.0 ceiling  -> score {score(1.0, BASE['S'] and 1.0, BASE['L']) if False else score(BASE['P'],1.0,BASE['L']):.4f}"
      f"  (max +{score(BASE['P'],1.0,BASE['L'])-b:.4f}); the SSIM axis alone cannot reach 78")
# balanced: equal thirds
th = need / 3.0
bP, bS, bL = BASE['P'] + th/0.6, BASE['S'] + th/30.0, BASE['L'] - th/40.0
print(f"BALANCED   : each axis supplies +{th:.4f}  ->  PSNR {bP:.4f} dB  SSIM {bS:.5f}  LPIPS {bL:.5f}")
print(f"             check {score(bP,bS,bL):.4f}")
print(f"             vs in-sample: PSNR {bP-INS['P']:+.3f} dB, SSIM {bS-INS['S']:+.4f}, LPIPS {bL-INS['L']:+.4f}")
# proportional (each metric improves by the same FRACTION of its own headroom to perfection)
# headroom: PSNR to 50, SSIM to 1, LPIPS to 0
def prop(f):
    return score(BASE['P'] + f*(50-BASE['P']), BASE['S'] + f*(1-BASE['S']), BASE['L']*(1-f))
lo, hi = 0.0, 1.0
for _ in range(80):
    m = (lo+hi)/2
    if prop(m) < TGT: lo = m
    else: hi = m
f = (lo+hi)/2
print(f"PROPORTIONAL (same fraction f of each metric's own headroom): f = {f:.5f}")
print(f"             PSNR {BASE['P']+f*(50-BASE['P']):.4f}  SSIM {BASE['S']+f*(1-BASE['S']):.5f}  "
      f"LPIPS {BASE['L']*(1-f):.5f}   -> {prop(f):.4f}")

# --- pricing the measured ceilings -----------------------------------------------------
print("\n=== PRICING EVERY MEASURED CEILING AGAINST THE +6.10 REQUIREMENT ===")
need_sr = TGT - score(**SR01)
items = [
    ("A1 per-frame sharpening oracle (spec grid, a>=0, GT-peeking)", +0.0193),
    ("A1 per-frame oracle over unsharp incl. NEGATIVE alpha (GT-peeking)", +0.2405),
    ("A1 oracle over ALL 30 operators tested (GT-peeking)", +0.2533),
    ("A1 best HONEST global operator (leave-one-out)", +0.0541),
    ("A1 spectral-match oracle (per-channel)", -1.8369),
    ("A1 MSE-optimal radial zero-phase filter (family LS optimum)", -0.7075),
    ("A2 oracle-keyed adaptive unsharp (uses actual GT sharpness)", +0.0170),
    ("A2 legal neighbour-predicted adaptive unsharp", +0.0100),
    ("A3 per-image photometric refit oracle (gain+bias per channel)", +0.1160),
    ("A3 best +-3px integer global re-registration", +0.0160),
    ("ENSEMBLE: 6-arm PNG mean vs best single (sr01)", 72.2644 - 71.9912),
    ("ENSEMBLE: mixsweep k=8 vs best single", 72.2527 - 71.9912),
    ("MODEL: scale_reg 0.01->0.1 (the one late-campaign WIN)", +0.192),
    ("PERFECT GENERALISATION, LPIPS only (0.2448 -> in-sample 0.2065)", 40*(0.2448-0.2065)),
    ("PERFECT GENERALISATION, full in-sample triple (74.697)", 74.697 - b),
    ("FIX the 8 sparse-coverage holes to the last-20 mean (75.99)", 75.9861 - 71.9912),
    ("FIX the 8 sparse-coverage holes to the best frame ever (81.98)", 77.7043 - 71.9912),
]
print(f"{'intervention':66s} {'dScore':>8s} {'% of +6.10':>11s}")
for n, d in items:
    print(f"{n:66s} {d:+8.4f} {100*d/need:10.1f}%")

# stacked best case
stack = 0.2533 + (72.2644-71.9912) + (74.697 - b)
print(f"\nOPTIMISTIC STACK (all-operator oracle + full ensemble + perfect generalisation, "
      f"assumed additive): {stack:+.4f} -> {b+stack:.4f}   still {TGT-(b+stack):.4f} SHORT")

# what the in-sample fit must become
gap = b - 0  # placeholder
insample_needed = TGT + (74.697 - b)
print(f"\nBINDING CONSTRAINT: held-out = in-sample - {74.697-b:.3f}. For held-out 78 the model must "
      f"fit its OWN training photos at {insample_needed:.2f}")
print(f"  (towers 80.8-81.8, chair 81.5, bonsai TODAY 74.70) -> bonsai must gain "
      f"+{insample_needed-74.697:.2f} IN-SAMPLE, i.e. the same +6.1, relocated to the fitting problem.")

# LB translation
print("\n=== LEADERBOARD TRANSLATION ===")
lb = 69.96
print(f"eval-split {b:.3f} reads {b-lb:.2f} HIGH vs the LB scene level {lb:.2f}")
print(f"eval 78 -> LB scene ~{78-(b-lb):.2f}; scene weight 1/7 -> total LB "
      f"+{(78-b)/7:.4f} at 1.0x transfer, +{0.85*(78-b)/7:.4f} at the 0.85x model-recipe rate")
print(f"For reference the ENTIRE campaign r28->r35 moved the LB +0.2077 in 8 rounds.")

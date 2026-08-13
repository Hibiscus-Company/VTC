---
name: uint8-deadband-operator-sign
description: "Sub-LSB image operators are silently destroyed by intermediate uint8 rounding; and fixing delivery is worthless (or harmful) until the operator's SIGN is verified per scene"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-28T07:59:24.815Z
---

Two coupled findings from the VAR2026 freeze day (28/07/2026), both measured.

**The deadband.** Any operator that writes an intermediate uint8 image and then adds a small
correction loses everything below half a level. `energy_restore` read `png_ens`, which stores
`round(mean)*255` — an exact integer per pixel — so the final write computed
`round(n + d) = n + round(d)` and every pixel with `|d| < 0.5` LSB got **zero** change.
Destroyed 49% of the intended correction on chair, 79% on bonsai. Public-harness cost at fixed
lambda: 1% at λ=1.0 (0.63 LSB) but **43% at λ=0.25** (0.16 LSB) — the damage is worst at low
amplitude, which is exactly where a low-disagreement ensemble pool operates. A harness validated
at high amplitude will never reveal it.
Fix: rebuild the mean in float from the dirs it was formed from and round **once**, at the end.
No re-rendering. Gate it by asserting `round(float mean)` reproduces the stored PNG except at
exact .5 ties (k=7 has no ties and is bit-exact; k=8 and 4:1:1 differ on ~3–6% of pixels by
exactly 1 LSB, and there the float mean is still exactly recoverable).

**The trap, which is the more valuable half.** Fixing delivery makes the operator *stronger*, so
it is only a gain where the operator's sign is positive. A λ sweep on real held-out GT showed the
same operator was **monotonically harmful on bonsai** (λ=0 → 71.955, 0.25 → 71.531, 1.0 → 70.760)
because that scene's members disagree from *reconstruction failure*, not from averaging loss — so
the disagreement map amplifies noise. The deadband had been accidentally protecting it. An audit
that prices *delivery* without checking *sign per scene* will confidently recommend amplifying a
harmful operator on the worst scene.

**Why:** these are both invisible to any composite-score check on a single well-behaved scene, and
the second one converts a correct bug-fix into a regression.

**How to apply:** before shipping any delivery/precision fix, (1) measure the delivered amplitude
in LSB, not the nominal parameter, and (2) sweep the operator's sign on every scene family that
will receive it. Ship it only where an independent measurement says the operator helps — for us
that meant towers only (LB-proven: λ=1.0 beat λ=0 by +0.1005), never bonsai.

Related: [[production-harness]] (the transfer rule that explains why the harness missed this),
[[ensemble-strategy]], [[bonsai-capacity-starvation]].

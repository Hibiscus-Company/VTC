#!/usr/bin/env python
"""Turn a measured HCM0181 tower-scene delta into an expected BLENDED leaderboard delta,
using only calibration constants already established in the campaign log.

  blended = tower_delta * (5 towers / 7 scenes) * REGIME * DEPTH
    REGIME  0.88  production-chain factor (the r28/r29 verifier's measured harness->LB factor;
                  r27 predicted its LB delta to 96%, r28's LPIPS transferred at 95%)
    DEPTH   marginal-value ratio between the harness pool size and the production pool size,
            taken from the measured ladder rather than assumed.
"""
import sys

def project(tower_delta, depth_ratio, regime=0.88, ntow=5, nsc=7, label=""):
    b = tower_delta * ntow / nsc * regime * depth_ratio
    print(f"{label:>34}  tower {tower_delta:+.4f}  x {ntow}/{nsc} x regime {regime} "
          f"x depth {depth_ratio:.3f}  =  blended {b:+.4f}")
    return b

if __name__ == "__main__":
    a = [float(x) for x in sys.argv[1:3]]
    project(a[0], a[1], label=sys.argv[3] if len(sys.argv) > 3 else "")

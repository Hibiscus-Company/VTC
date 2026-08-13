#!/usr/bin/env python
"""Build the combined pass-2 arm spec. One pool (P8), zero weights select the sub-pool, so both
the family-weight question and the 8-member LS question ride on ONE set of member loads."""
import json
import numpy as np
from a1_solve import NAMES, ls_weights
from a2_family import IX, ev, odd

UT = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
P8 = UT + ["m31b_taillpips", "e17visnorm", "e15ceil95", "gsplatB7ppisp2"]
i8 = [IX[n] for n in P8]
i5 = [IX[n] for n in UT + ["m31b_taillpips"]]

w5e = ls_weights(i5, sel=ev)          # fit on EVEN images only -> honest on odd images
w8e = ls_weights(i8, sel=ev)


def pad(w5):
    return list(w5) + [0.0, 0.0, 0.0]


def fam(wB):
    return pad(list(np.full(4, (1 - wB) / 4)) + [wB])


arms = [
    dict(name="uniform5_CONTROL",  w=fam(0.2)),
    dict(name="wB0.273_prod1.5x",  w=fam(1.5 / 5.5)),
    dict(name="wB0.273_frozenC",   w=fam(1.5 / 5.5), corr=1.25),   # restore-coupling control
    dict(name="UT4_only_wB0",      w=fam(0.0)),
    dict(name="LS5_heldout",       w=pad(list(w5e))),
    dict(name="uniform8",          w=[1 / 8.] * 8),
    dict(name="LS8_heldout",       w=list(w8e)),
]
json.dump(dict(members=P8, arms=arms),
          open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wtq/arms_MAIN.json", "w"), indent=1)
for a in arms:
    print(f"{a['name']:>18}", [round(x, 4) for x in a["w"]], "corr=", a.get("corr"))

| scene | member | seed | recipe note | size |
|---|---|---|---|---|
| bonsai | `r14/bonsai_aa42` | seed 42 | r14 recipe (pre scale_reg) | 1.10 GB |
| bonsai | `r14/bonsai_aa7` | seed 7 | r14 recipe (pre scale_reg) | 1.10 GB |
| bonsai | `r14/bonsai_aa13` | seed 13 | r14 recipe (pre scale_reg) | 1.10 GB |
| chair | `r14/chair_aa42` | seed 42 | r14 recipe | 1.76 GB |
| chair | `r14/chair_aa7` | seed 7 | r14 recipe | 1.76 GB |
| chair | `r14/chair_aa13` | seed 13 | r14 recipe | 1.76 GB |
| chair | `r17/chair_ema099_seed42` | seed 42 | --ema_decay 0.99 | 1.76 GB |
| chair | `r17/chair_ema099_seed7` | seed 7 | --ema_decay 0.99 | 1.76 GB |
| chair | `r17/chair_ema099_seed13` | seed 13 | --ema_decay 0.99 | 1.76 GB |
| chair | `r17/chair_depth_seed42` | seed 42 | depth variant | 1.76 GB |
| HCM0644 | `r28_members/HCM0644` | seed 42 | shipped tower recipe | 1.76 GB |
| HCM0674 | `r28_members/HCM0674` | seed 42 | shipped tower recipe | 1.76 GB |

**12 checkpoints, 19.1 GB total.**

| scene | members whose weights are GONE | count |
|---|---|---|
| bonsai | `r24_bonsai/aa101 / aa202 / aa303` | 3 |
| bonsai | `r28_members/bonsai` | 1 |
| bonsai | `r38_prod/s111 s555 s777 s222 s333 s999  (--scale_reg 0.1)` | 6 |
| chair | `r28_members/chair` | 1 |
| HCM0421 | `r22 chain (r20 base + seed101) / r25_mip3d / r28_members` | 3 |
| HCM0539 | `r22 chain / r25_mip3d / r28_members` | 3 |
| HCM0540 | `r22 chain / r25_mip3d / r28_members` | 3 |
| HCM0644 | `r22 chain / r25_mip3d` | 2 |
| HCM0674 | `r22 chain / r25_mip3d` | 2 |

**24 of the 31 members no longer have weights.**

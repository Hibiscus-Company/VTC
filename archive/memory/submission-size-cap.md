---
name: submission-size-cap
description: "The VAR competition's 350MB submission cap is 350 MiB (367,001,600 bytes), not 350e6 — user-confirmed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-27T08:31:48.791Z
---

The submission zip cap is **350 MiB = 350×1024×1024 = 367,001,600 bytes**, NOT 350,000,000.
User stated this explicitly on 2026-07-27.

We had enforced the decimal reading in `scripts/verify_zip.py` since round 1, which silently
threw away ~17 MB of budget. Fixed on 2026-07-27 — the verifier now measures MiB and prints
both readings.

**Why:** byte budget converts directly into score. It is what caps the energy-restoration
lambda, and it forced HCM0421 down to JPEG Q99 for several rounds. r27 sits at 329.11 MiB,
so real headroom was 20.89 MiB where I had been budgeting 4.91 MB.

**CONFIRMED EMPIRICALLY 2026-07-27:** r28 shipped at 357,078,727 bytes (340.54 MiB, i.e. 357.08
MB decimal) and was ACCEPTED and graded (77.5029). The MiB reading is correct; the decimal
companion build was unnecessary.

**How to apply:** budget in MiB, up to 367,001,600 bytes. No need for decimal-safe companion
builds any more.

Related: [[nvs-competition-setup]], [[production-harness]]

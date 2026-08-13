#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
G=/home/bkai/.claude/jobs/1c9cf7e9/tmp/lens/member_gate.py
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  PEERS="/mnt/d/avv/r2r9/models/${T}_ut42/test_png /mnt/d/avv/r2r9/models/${T}_ut7/test_png /mnt/d/avv/r22_seed101/${T}/test_png /mnt/d/avv/r25_mip3d/${T}/test_png"
  for C in /mnt/d/avv/output_s2gates/${T}_champA/test_png \
           /mnt/d/avv/output_s2gates/${T}_memB/test_png \
           /mnt/d/avv/output_s2gates/${T}_memC/test_png \
           /mnt/d/avv/r17/${T}_ut7_ema999/test_png \
           /mnt/d/avv/r2r8/models/${T}_ut42/test_png \
           /mnt/d/avv/r2r8/models/${T}_ut7/test_png; do
    [ -d "$C" ] || continue
    echo "### $T :: $(echo $C | sed 's#/mnt/d/avv/##;s#/test_png##')"
    python $G --candidate "$C" --peers $PEERS --expect 60 2>&1 | grep -vE "Warning|warn" | tail -4
  done
done

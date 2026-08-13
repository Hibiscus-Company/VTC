#!/bin/bash
# Wait for exp12 champion validation, then build full 13-scene submission
until grep -q "CHAMP DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/queue8.log 2>/dev/null; do sleep 60; done
bash /mnt/c/Users/BKAI/an_plaza2/FastGS/build_submission_champ.sh round1_champ

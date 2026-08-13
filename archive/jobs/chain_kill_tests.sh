#!/bin/bash
# waits for the gates queue to exit, then fires kill_tests.sh immediately — no idle gap
while pgrep -f "gates_queue.sh" >/dev/null; do sleep 60; done
sleep 30   # let renders/filesystem settle
setsid nohup bash /home/bkai/.claude/jobs/1c9cf7e9/tmp/kill_tests.sh > /mnt/d/avv/kill_tests.log 2>&1 &
echo "kill_tests launched $(date)"

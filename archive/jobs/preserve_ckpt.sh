#!/bin/bash
# RACE THE DELETER. prod_long45k.sh does: train -> render -> census -> rm -f ckpt.pt
# That leaves a window of several minutes between the checkpoint appearing and being deleted.
# The organiser now requires the winning weights to be submitted, and r36's members were already
# destroyed by that same rm. These two are the r37 (long45k) members -- do not lose them too.
DEST=/mnt/d/avv/WEIGHTS; mkdir -p $DEST
declare -A got
while true; do
  for M in /mnt/d/avv/r45_prod/s202 /mnt/d/avv/r45_prod/s303; do
    tag=$(basename $M)
    [ -n "${got[$tag]}" ] && continue
    if [ -f "$M/ckpt.pt" ]; then
      free=$(df -Pm /mnt/d | awk 'NR==2{print $4}')
      if [ "$free" -lt 3000 ]; then echo "$(date +%H:%M) !! only ${free}MB free, refusing to copy $tag"; continue; fi
      cp -f "$M/ckpt.pt" "$DEST/bonsai_long45k_$tag.pt" && got[$tag]=1 \
        && echo "$(date +%H:%M) preserved $tag -> $DEST/bonsai_long45k_$tag.pt ($(du -h $DEST/bonsai_long45k_$tag.pt|cut -f1))"
    fi
  done
  [ -n "${got[s202]}" ] && [ -n "${got[s303]}" ] && break
  sleep 20
done
echo "=== both r37 member checkpoints preserved $(date +%H:%M) ==="; touch /mnt/d/avv/WEIGHTS/PRESERVED.DONE

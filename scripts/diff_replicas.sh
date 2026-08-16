#!/usr/bin/env bash
# Show how each replica has diverged from canonical original/ (works without git).
cd "$(dirname "$0")/.."
for P in an bach tu; do
  echo "=== $P vs original ==="
  diff -rq --exclude "__pycache__" original "$P" 2>/dev/null | sed "s/^/  /" || true
done
echo
echo "Promote a change:  cp <replica>/<file> original/<file>   then rerun make_replicas.sh"

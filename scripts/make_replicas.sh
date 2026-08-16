#!/usr/bin/env bash
# Regenerate the per-teammate working copies from canonical original/.
# WARNING: overwrites an/ bach/ tu/ — run diff_replicas.sh first and promote
# anything worth keeping into original/.
set -e
cd "$(dirname "$0")/.."
for P in an bach tu; do
  rsync -a --delete --exclude "__pycache__" original/ "$P/"
  echo "replica $P/ <- original/ ($(ls original | wc -l) files)"
done

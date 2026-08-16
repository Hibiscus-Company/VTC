#!/usr/bin/env bash
# Produce dist/onsite_bundle.zip — exactly what gets uploaded into the competition
# Jupyter environment. Code + docs + configs + env sources. Datasets and runs never.
#   bash scripts/make_bundle.sh [--with-weights] [--with-archive] [--with-wheels]
set -e
cd "$(dirname "$0")/.."
mkdir -p dist
OUT=dist/onsite_bundle.zip; rm -f "$OUT"
INC=(original an bach tu docs configs scripts notebooks env/requirements.txt \
     env/SETUP.md env/build_extensions.sh env/fetch_weights.sh env/vendor \
     CLAUDE.md README.md LICENSE LICENSE_ORIGINAL.md dataset/README.md)
for a in "$@"; do case $a in
  --with-weights) INC+=(env/weights);;
  --with-archive) INC+=(archive);;
  --with-wheels)  INC+=(env/wheels);;
esac; done
zip -qr "$OUT" "${INC[@]}" -x "*__pycache__*" -x "*.pyc" -x "*.ipynb_checkpoints*"
python3 - <<PY
import zipfile, os
z = zipfile.ZipFile("$OUT")
print(f"{len(z.namelist())} files, {os.path.getsize('$OUT')/2**20:.1f} MiB -> $OUT")
PY
echo "Upload this zip; on-site: unzip, then follow env/SETUP.md section A."

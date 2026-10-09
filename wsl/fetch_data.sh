#!/usr/bin/env bash
# Downloads the Tanks&Temples/DeepBlending COLMAP data (official 3DGS release) and extracts the truck scene.
set -e
ROOT="/mnt/d/College/3D Vision"
mkdir -p "$ROOT/data"
cd "$ROOT/data"
if [ ! -d tandt/truck ]; then
  [ -f tandt_db.zip ] || wget -q --show-progress -O tandt_db.zip https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
  unzip -q -o tandt_db.zip 'tandt/truck/*' -d .
fi
ls tandt/truck tandt/truck/sparse/0
echo FETCH_DONE

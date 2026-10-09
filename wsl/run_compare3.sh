#!/usr/bin/env bash
# DGE vs Edit3D (first run) vs Edit3D with the ADSS fixes, same cameras, masks from the fixed run.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
ROOT="/mnt/d/College/3D Vision"
OUT="$ROOT/outputs"
cd "$ROOT/DGE"
V1=$(ls "$OUT"/edit3d_red_truck/outputs/edit-n2n/*/save/last.ply | head -1)
V2=$(ls "$OUT"/full_v2/outputs/edit-n2n/*/save/last.ply | head -1)
echo "Edit3D v1 model: $V1"; echo "Edit3D fixed model: $V2"
python tools/compare_models.py "$ROOT/data/tandt/truck" \
  "$OUT/truck_lean/point_cloud/iteration_15000/point_cloud.ply" \
  "$OUT/full_v2/results/truck_red_v2/masks" \
  "$OUT/compare3_red_truck.png" "$OUT/compare3_red_truck.json" 4 \
  "DGE=$OUT/dge/dge/Turn_the_truck_into_a_red_truck@20261005-004147/save/last.ply" \
  "Edit3D released=$V1" "Edit3D fixed=$V2" 2>&1 | grep -vE "Warning|warn|Reading camera"

#!/usr/bin/env bash
# Compare DGE vs Edit3D on the red-truck instruction, same cameras, Edit3D's masks.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
ROOT="/mnt/d/College/3D Vision"
E3D="$ROOT/outputs/edit3d_red_truck"
cd "$ROOT/DGE"
python tools/compare_edit3d_dge.py "$ROOT/data/tandt/truck" \
  "$ROOT/outputs/truck_lean/point_cloud/iteration_15000/point_cloud.ply" \
  "$ROOT/outputs/dge/dge/Turn_the_truck_into_a_red_truck@20261005-004147/save/last.ply" \
  "$E3D/outputs/edit-n2n/Turn_the_truck_into_a_red_truck@20261005-103857/save/last.ply" \
  "$E3D/results/truck_red/masks" "$ROOT/outputs/compare_dge_vs_edit3d_red_truck.png" \
  "$ROOT/outputs/compare_dge_vs_edit3d_red_truck.json" 4 2>&1 | grep -vE "Warning|warn|Reading camera"

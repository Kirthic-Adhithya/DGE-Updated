#!/usr/bin/env bash
# Evaluate all edited truck models against the original model (same 48 views, same depth, same metric).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
ROOT="/mnt/d/College/3D Vision"
R="$ROOT/outputs/dge/dge/Turn_it_into_a_snowy_winter_scene_with_thick_snow_covering_the_truck_and_the_ground@2026100"
cd "$ROOT/DGE"
python tools/eval_consistency.py --source "$ROOT/data/tandt/truck" \
    --orig "$ROOT/outputs/truck_lean/point_cloud/iteration_15000/point_cloud.ply" \
    --edited baseline="${R}4-161200/save/last.ply" baseline_repeat="${R}4-180334/save/last.ply" \
             ours_v1="${R}4-173923/save/last.ply" ours_v2="${R}4-175243/save/last.ply" \
    --views 48 --out "$ROOT/outputs/eval_consistency_v2.json" --save_dir "$ROOT/outputs/eval_maps" 2>&1 \
    | grep -vE "Warning|warn|Reading camera"

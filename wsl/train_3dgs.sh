#!/usr/bin/env bash
# Train the source 3DGS (input to DGE editing) with train_3dgs.py.
# Usage: bash wsl/train_3dgs.sh [model_name] [densify_grad_threshold]
#   truck_3dgs  = default threshold 2e-4  (~2.2M Gaussians, too big to edit on 8 GB)
#   truck_lean  = threshold 6e-4          (far fewer Gaussians, fits DGE editing on 8 GB)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
ROOT="/mnt/d/College/3D Vision"
NAME="${1:-truck_3dgs}"
THR="${2:-2e-4}"
cd "$ROOT/DGE"
python tools/train_3dgs.py -s "$ROOT/data/tandt/truck" -m "$ROOT/outputs/$NAME" \
    --iterations 15000 --densify_grad_threshold "$THR" --data_device cpu --eval > "$HOME/train_$NAME.log" 2>&1
echo "EXIT_CODE=$?" >> "$HOME/train_$NAME.log"

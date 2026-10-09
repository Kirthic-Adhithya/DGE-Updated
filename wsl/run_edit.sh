#!/usr/bin/env bash
# Run a DGE edit on the trained truck 3DGS.  Usage: bash wsl/run_edit.sh "<instruction>" [extra hydra-style overrides...]
# Logs to ~/edit.log. The SD-1.5 repo under runwayml/ was removed from the Hub, so we use the official mirror.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
export HF_HUB_DISABLE_TELEMETRY=1
# note: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments on WSL2 -- it causes "CUDA driver error: device not ready"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.7
export PYTHONUNBUFFERED=1   # show progress prints live in the log
ROOT="/mnt/d/College/3D Vision"
PROMPT="${1:-Make it look like winter, covered in snow}"
shift || true
cd "$ROOT/DGE"
python launch.py --config configs/dge.yaml --train --gpu 0 \
    exp_root_dir="$ROOT/outputs/dge" \
    data.source="$ROOT/data/tandt/truck" \
    system.gs_source="$ROOT/outputs/${MODEL:-truck_lean}/point_cloud/iteration_15000/point_cloud.ply" \
    system.prompt_processor.prompt="$PROMPT" \
    system.prompt_processor.pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    "$@" > "$HOME/edit.log" 2>&1
echo "EXIT_CODE=$?" >> "$HOME/edit.log"

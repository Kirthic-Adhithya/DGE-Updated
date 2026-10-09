#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export HF_HUB_DISABLE_TELEMETRY=1
cd "$(dirname "$0")/.."
python tools/predownload_models.py > "$HOME/predownload.log" 2>&1
echo "EXIT_CODE=$?" >> "$HOME/predownload.log"

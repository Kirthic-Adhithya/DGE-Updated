#!/usr/bin/env bash
# Run a python script from the DGE repo root inside the DGE env.  Usage: bash wsl/run_py.sh script.py [args...]
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cd "$(dirname "$0")/.."
python "$@" 2>&1 | grep -v -E "Warning|warn"

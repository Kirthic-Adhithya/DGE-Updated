#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cd "$(dirname "$0")/.."
python tools/bench_epipolar.py 2>&1 | grep -v Warning | tail -8

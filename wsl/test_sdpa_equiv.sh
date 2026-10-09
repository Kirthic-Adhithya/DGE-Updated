#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
# stop any running edit first (by pid, so we don't match this shell)
for p in $(ps -eo pid,cmd | grep '[p]ython launch.py' | awk '{print $1}'); do kill -9 "$p"; done
cd "$(dirname "$0")/.."
python tools/test_sdpa_equiv.py

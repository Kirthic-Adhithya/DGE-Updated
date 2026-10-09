#!/usr/bin/env bash
# Dump the current Python stack of the running DGE edit (needs: pip install py-spy)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
PID=$(ps -eo pid,cmd | grep '[p]ython launch.py' | awk '{print $1}' | head -1)
echo "pid=$PID"
for i in 1 2 3; do
  py-spy dump --pid "$PID" 2>&1 | head -24 | cut -c1-160
  echo -----
  sleep 4
done

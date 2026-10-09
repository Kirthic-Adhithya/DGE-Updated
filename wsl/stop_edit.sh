#!/usr/bin/env bash
# Kill any running DGE edit by pid (pkill -f would also match the calling shell).
for p in $(ps -eo pid,cmd | grep '[p]ython launch.py' | awk '{print $1}'); do kill -9 "$p"; done
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader

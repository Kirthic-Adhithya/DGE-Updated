#!/usr/bin/env bash
# Show last [DGE] progress lines and the first traceback's repo frames + final error from ~/edit.log
LOG="${1:-$HOME/edit.log}"
grep -E '\[DGE\]|Editing finished' "$LOG" | tail -3 | cut -c1-170
echo "--- first traceback (repo frames + errors) ---"
tr '\r' '\n' < "$LOG" | grep -vE 'it/s\]|^\s*$' | awk '/Traceback/{c++} c==1' | grep -E 'File "/mnt|Error' | cut -c1-200
echo "--- dmesg ---"
dmesg 2>/dev/null | grep -iE 'dxg|make_resident' | tail -3 | cut -c1-160

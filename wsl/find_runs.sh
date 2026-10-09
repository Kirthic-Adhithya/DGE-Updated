#!/usr/bin/env bash
# List DGE run folders (newest last) with whether they contain a final last.ply
grep -E 'EXIT_CODE|Error' ~/edit.log | tail -2
ls -dt /mnt/d/College\ 3D\ Vision/outputs/dge/dge/*/ | tac | while read d; do
  [ -f "$d/save/last.ply" ] && echo "OK   $(basename "$d")" || echo "none $(basename "$d")"
done

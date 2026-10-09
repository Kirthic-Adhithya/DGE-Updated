#!/usr/bin/env bash
# DGE edit on the lab machine. Paths are RELATIVE to the repo root (this repo is cloned next to the data):
#
#   <parent>/DGE/          <- this repo
#   <parent>/truck/        <- COLMAP scene (images/, sparse/0/)
#   <parent>/truck_lean.ply  <- trained 3DGS of the scene (any 3DGS .ply works; override with PLY=...)
#
# Usage:  bash lab/run_dge_lab.sh "Turn the truck into a red truck" 2>&1 | tee dge_red.log
# Env knobs: DATA=../truck  PLY=../truck_lean.ply  OUT=../outputs_dge  CACHE=../KG_cache
# Extra hydra overrides can be appended:  bash lab/run_dge_lab.sh "prompt" data.max_view_num=16
set -u
cd "$(dirname "$0")/.."
DATA=${DATA:-../truck}; PLY=${PLY:-../truck_lean.ply}; OUT=${OUT:-../outputs_dge}; CACHE=${CACHE:-../KG_cache}
PROMPT="${1:-Turn the truck into a red truck}"; shift || true

fail=0
[ -d "$DATA/images" ]   || { echo "MISSING: $DATA/images"; fail=1; }
[ -d "$DATA/sparse/0" ] || { echo "MISSING: $DATA/sparse/0"; fail=1; }
[ -f "$PLY" ]           || { echo "MISSING: $PLY"; fail=1; }
[ "$fail" = 0 ] || { echo "preflight failed (see the layout at the top of this script)"; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate DGE
mkdir -p "$CACHE/hf" "$CACHE/torch" "$OUT"
export HF_HOME="$(cd "$CACHE/hf" && pwd)" TORCH_HOME="$(cd "$CACHE/torch" && pwd)" XDG_CACHE_HOME="$(cd "$CACHE" && pwd)"
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.7

python launch.py --config configs/dge.yaml --train --gpu 0 \
  exp_root_dir="$OUT" \
  data.source="$DATA" \
  system.gs_source="$PLY" \
  system.prompt_processor.prompt="$PROMPT" \
  system.prompt_processor.pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  "$@"

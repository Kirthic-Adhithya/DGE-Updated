#!/usr/bin/env bash
# Two back-to-back runs with the SAME prompt/settings (seed 0):
#   1) ours v2  : consistency-weighted fitting with misalignment tolerance (sigma 0.05)
#   2) baseline2: plain DGE again -> run-to-run noise floor for the metrics
cd "$(dirname "$0")/.."
PROMPT="Turn it into a snowy winter scene with thick snow covering the truck and the ground"
COMMON="system.guidance.guidance_scale=12.5 system.guidance.camera_batch_size=4 system.guidance.vae_chunk_size=2 data.max_view_num=16"
bash wsl/run_edit.sh "$PROMPT" $COMMON system.use_consistency_weight=true system.cw_sigma=0.05 system.cw_min_weight=0.05
cp ~/edit.log ~/edit_ours_v2.log
bash wsl/run_edit.sh "$PROMPT" $COMMON
cp ~/edit.log ~/edit_baseline2.log
echo EXPERIMENTS_DONE > ~/experiments.done

# Running this fork on the lab machine

This is DGE (Chen, Laina, Vedaldi, ECCV 2024) plus our changes. Upstream: https://github.com/silent-chen/DGE

## What we changed
- `threestudio/utils/dge_utils.py`: fused attention (`scaled_dot_product_attention`) in both attention paths. Results are identical (max diff 8e-16) and the memory peak is about half.
- `threestudio/models/guidance/dge_guidance.py`: chunked VAE encode/decode (`vae_chunk_size`), per-step timing and memory prints.
- `threestudio/systems/DGE.py`: lazy SAM loading, original frames kept on the CPU, optional consistency-weighted fitting (`use_consistency_weight`; tested, no measurable gain).
- `threestudio/utils/consistency.py`: depth-warp based inconsistency measure.
- `tools/`: Python helpers (`compare_models.py`, `eval_consistency.py`, `train_3dgs.py`, tests, benchmarks). Run from the repo root, e.g. `python tools/compare_models.py ...`.
- `lab/`: Linux scripts with relative paths, for the lab machine.
- `wsl/`: shell scripts written for the WSL laptop; they contain `/mnt/d/...` paths, so ignore them on the lab machine.

## Steps on the lab machine
```bash
git clone git@github.com:<you>/<repo>.git DGE       # next to the truck/ folder
cd DGE
bash lab/setup_lab.sh 2>&1 | tee setup_lab.log      # once, ~30-40 min
bash lab/run_dge_lab.sh "Turn the truck into a red truck" 2>&1 | tee dge_red.log
```
Models (about 5 GB: InstructPix2Pix, Stable Diffusion 1.5, SAM) download on the first run into `../KG_cache`.
Output goes to `../outputs_dge/`; the edited model is `.../save/last.ply`.
On a 24 GB card you can raise `system.guidance.camera_batch_size` and `data.max_view_num` (defaults in `configs/dge.yaml`).

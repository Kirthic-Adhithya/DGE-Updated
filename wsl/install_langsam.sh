#!/usr/bin/env bash
# lang-segment-anything is imported unconditionally by threestudio/utils/sam.py, which uses the
# pre-SAM2 API (LangSAM("vit_h").predict -> 4-tuple). a1a9557 is the last such commit (2024-04-12).
# --no-deps so its old pins (Pillow 9.3, gradio 3, hub 0.16) can't disturb the env.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9" LIBRARY_PATH="/usr/lib/wsl/lib"
C="-c $HOME/constraints.txt"
pip install --no-deps --no-build-isolation "git+https://github.com/luca-medeiros/lang-segment-anything.git@a1a9557"
pip install $C --no-deps git+https://github.com/facebookresearch/segment-anything.git
pip install $C --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git 2>&1 | tail -15 | cut -c1-200
pip install $C "numpy<2" supervision==0.18.0 addict yapf timm pycocotools 2>&1 | tail -3 | cut -c1-200

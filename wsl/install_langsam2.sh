#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH" LIBRARY_PATH="/usr/lib/wsl/lib"
C="-c $HOME/constraints.txt"
pip install $C poetry-core "supervision==0.22.0"
pip install $C --no-deps --no-build-isolation "git+https://github.com/luca-medeiros/lang-segment-anything.git@a1a9557" 2>&1 | tail -3
cd "/mnt/d/College/3D Vision/DGE"
python -c "
import threestudio
print('threestudio import OK')
" 2>&1 | tail -8

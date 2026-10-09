#!/usr/bin/env bash
# Builds tiny-cuda-nn in the DGE env. WSL keeps libcuda.so in /usr/lib/wsl/lib, which the linker must be told about.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9" TCNN_CUDA_ARCHITECTURES=89 MAX_JOBS=4
export LIBRARY_PATH="/usr/lib/wsl/lib:$LIBRARY_PATH" LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" > ~/tcnn.log 2>&1
echo "EXIT_CODE=$?" >> ~/tcnn.log

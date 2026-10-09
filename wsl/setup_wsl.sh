#!/usr/bin/env bash
# One-shot environment setup for DGE inside WSL2 (Ubuntu 22.04) on an RTX 4060 (sm_89).
# Usage (inside WSL):  bash wsl/setup_wsl.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root: the CUDA extensions below are installed from gaussiansplatting/

ENV_NAME=DGE
CONDA_DIR="$HOME/miniconda3"

if [ ! -d "$CONDA_DIR" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

conda env list | grep -q "^$ENV_NAME " || conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# CUDA 11.8 toolkit (nvcc) so the extensions can be compiled; no sudo needed.
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=4

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" ninja cmake "setuptools<70" wheel   # torch cpp_extension needs pkg_resources

# Pins needed because diffusers 0.19.3 predates newer hub/transformers APIs.
pip install "huggingface_hub==0.19.4" "transformers==4.36.2" "accelerate==0.25.0" "diffusers[torch]==0.19.3"

# Core deps (requirements.txt minus the heavy/optional ones handled below)
pip install omegaconf==2.3.0 jaxtyping typeguard nerfacc==0.3.3 opencv-python matplotlib \
    "imageio>=2.28.0" "imageio[ffmpeg]" "libigl==2.4.1" xatlas "trimesh[easy]" networkx pysdf PyMCubes wandb \
    tensorboard==2.16.2 torchmetrics einops kornia==0.7.0 torch_efficient_distloss mediapy "plyfile<1.1" rembg scikit-learn scipy \
    viser easydict "albumentations==0.5.2" webdataset pyrender "lightning==2.0.9" "pytorch-lightning==2.0.9" \
    lpips
pip install "numpy<2" "opencv-python<4.10" "protobuf<5"
# git packages that compile or import torch at build time need --no-build-isolation
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-build-isolation git+https://github.com/openai/CLIP.git git+https://github.com/ashawkey/envlight.git
pip install "numpy<2" "opencv-python<4.10" "opencv-python-headless<4.10" "protobuf<5" "huggingface_hub==0.19.4" "setuptools<70"

# Gaussian-splatting CUDA extensions
pip install --no-build-isolation gaussiansplatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation gaussiansplatting/submodules/simple-knn

# tiny-cuda-nn (slow compile, ~10+ min). WSL keeps libcuda.so in /usr/lib/wsl/lib.
export LIBRARY_PATH="/usr/lib/wsl/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES=89
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

echo "SETUP DONE"

# --- steps added after the first successful run (kept in separate scripts, run them in this order) ---
#   bash wsl/install_langsam.sh && bash wsl/install_langsam2.sh   # lang-sam pre-SAM2 commit + segment-anything + GroundingDINO
#   python tools/predownload_models.py                           # InstructPix2Pix / SD-1.5 weights with retry+resume
#   bash wsl/train_3dgs.sh truck_lean 6e-4                     # source 3DGS (fits DGE editing in 8 GB)
#   bash wsl/run_edit.sh "<instruction>" system.guidance.camera_batch_size=4 system.guidance.vae_chunk_size=2 data.max_view_num=16

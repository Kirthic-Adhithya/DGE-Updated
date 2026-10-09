#!/usr/bin/env bash
# One-time environment setup for DGE on a Linux lab machine (RTX A5000 = compute capability 8.6).
# Run from anywhere:  bash lab/setup_lab.sh 2>&1 | tee setup_lab.log      (takes ~30-40 min, tiny-cuda-nn compiles slowly)
# Needs: conda (miniconda) on PATH or in ~/miniconda3, an NVIDIA driver, and internet.
# Everything is installed into a conda env called DGE; nothing outside it is touched.
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root
ARCH=${ARCH:-8.6}                             # 8.6 = A5000/3090, 8.9 = 4090/4060, 8.0 = A100
CONDA_SH=${CONDA_SH:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh}
source "$CONDA_SH"

conda env list | grep -q "^DGE " || conda create -n DGE python=3.10 -y
conda activate DGE

# CUDA 11.8 toolkit (nvcc) inside the env so the CUDA extensions compile; no sudo needed.
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="$ARCH" TCNN_CUDA_ARCHITECTURES="${ARCH/./}" MAX_JOBS=4

# pins that keep the old diffusers 0.19.3 working; applied to every later install
cat > "$CONDA_PREFIX/constraints.txt" <<'PINS'
numpy<2
protobuf<5
setuptools<70
huggingface_hub==0.19.4
transformers==4.36.2
opencv-python<4.10
opencv-python-headless<4.10
PINS
export PIP_CONSTRAINT="$CONDA_PREFIX/constraints.txt"

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install ninja cmake wheel
pip install "accelerate==0.25.0" "diffusers[torch]==0.19.3"
pip install omegaconf==2.3.0 jaxtyping typeguard nerfacc==0.3.3 opencv-python matplotlib \
    "imageio>=2.28.0" "imageio[ffmpeg]" "libigl==2.4.1" xatlas "trimesh[easy]" networkx pysdf PyMCubes wandb \
    tensorboard==2.16.2 torchmetrics einops kornia==0.7.0 torch_efficient_distloss mediapy "plyfile<1.1" rembg \
    scikit-learn scipy viser easydict "albumentations==0.5.2" webdataset pyrender \
    "lightning==2.0.9" "pytorch-lightning==2.0.9" lpips

# packages that import torch while building need --no-build-isolation
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-build-isolation git+https://github.com/openai/CLIP.git git+https://github.com/ashawkey/envlight.git
pip install --no-build-isolation gaussiansplatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation gaussiansplatting/submodules/simple-knn
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# lang-sam (threestudio/utils/sam.py imports it unconditionally; a1a9557 = last commit before the SAM2 API change)
pip install poetry-core "supervision==0.18.0" addict yapf timm pycocotools
pip install --no-deps --no-build-isolation "git+https://github.com/luca-medeiros/lang-segment-anything.git@a1a9557"
pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

python -c "import torch, threestudio; print('OK  torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "SETUP DONE"

#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate DGE
pip install -q -c "$HOME/constraints.txt" "tensorboard==2.16.2" "protobuf<5" 2>&1 | tail -2
python -c "from torch.utils.tensorboard import SummaryWriter; print('tb ok')" 2>&1 | tail -1

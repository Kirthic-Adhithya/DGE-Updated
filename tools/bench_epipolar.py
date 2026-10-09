"""Micro-benchmark of compute_epipolar_constrains and its parts, using synthetic cameras (only needs full_proj_transform)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import time
from types import SimpleNamespace

import torch

from gaussiansplatting.utils.graphics_utils import get_fundamental_matrix_with_H
from threestudio.utils.dge_utils import compute_epipolar_constrains, point_to_line_dist

torch.manual_seed(0)


def cam():
    return SimpleNamespace(full_proj_transform=(torch.eye(4) + 0.1 * torch.randn(4, 4)).cuda())


def timeit(fn, n=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


c1, c2 = cam(), cam()
for H in (64, 32, 16, 8):
    full = timeit(lambda: compute_epipolar_constrains(c1, c2, current_H=H, current_W=H))
    fmat = timeit(lambda: get_fundamental_matrix_with_H(c1, c2, H, H))
    print(f"H={H:3d}: full call {full:6.1f} ms | fundamental matrix part {fmat:6.1f} ms")

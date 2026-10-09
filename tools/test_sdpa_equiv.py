"""Sanity checks that fused SDPA equals the original attention code in dge_utils.py.

1. extended attention: per-head bmm/softmax loop vs SDPA
2. normal attention (get_attention_scores + bmm) vs SDPA on the (batch*heads, seq, dim) layout
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import torch

torch.manual_seed(0)
n, h, s, d = 3, 8, 64, 40  # frames, heads, tokens, head dim (small sizes; shapes mirror dge_utils.py)
sdpa = torch.nn.functional.scaled_dot_product_attention
scale = d**-0.5

# 1. extended attention
q = torch.randn(n, h, s, d, dtype=torch.float64)
k = torch.randn(n, h, s * n, d, dtype=torch.float64)
v = torch.randn(n, h, s * n, d, dtype=torch.float64)
outs = []
for j in range(h):
    sim = torch.bmm(q[:, j], k[:, j].transpose(-1, -2)) * scale
    outs.append(torch.bmm(sim.softmax(dim=-1), v[:, j]))
old = torch.cat(outs, dim=0).view(h, n, s, d).permute(1, 0, 2, 3).reshape(h * n, s, -1)
new = sdpa(q, k, v).reshape(n * h, s, -1)
print("extended attention max abs diff:", (old - new).abs().max().item())
assert torch.allclose(old, new, atol=1e-10)

# 2. normal attention on (batch*heads, seq, dim) -- diffusers' get_attention_scores is baddbmm(q, k^T, scale) -> softmax
qb, kb, vb = (torch.randn(n * h, s, d, dtype=torch.float64) for _ in range(3))
probs = torch.baddbmm(torch.empty(n * h, s, s, dtype=torch.float64), qb, kb.transpose(-1, -2), beta=0, alpha=scale).softmax(dim=-1)
old2 = torch.bmm(probs, vb)
new2 = sdpa(qb, kb, vb)
print("normal attention max abs diff:", (old2 - new2).abs().max().item())
assert torch.allclose(old2, new2, atol=1e-10)
print("EQUIVALENT")

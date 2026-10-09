"""Synthetic check of threestudio/utils/consistency.py: a textured plane seen from two cameras with known geometry.

Expectations:
  * warp(A -> B) of the true plane image reproduces A's pixels (error ~0)  -> conventions/geometry are right
  * an edit that is consistent across views gets ~zero inconsistency
  * an edit that paints a view-specific blob gets high inconsistency exactly there, and a low weight
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import math
from types import SimpleNamespace

import torch

from threestudio.utils.consistency import edit_inconsistency, inconsistency_to_weight, warp_to_view

dev = "cuda"
H = W = 128
FOV = math.radians(60)


def make_cam(yaw_deg, cx=0.0):
    a = math.radians(yaw_deg)
    R = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=torch.float32)
    t = torch.tensor([cx, 0.0, 3.0])
    Wc = torch.eye(4)
    Wc[:3, :3] = R
    Wc[:3, 3] = t  # the world origin sits 3 units in front of the camera
    Wr = Wc.T.contiguous().to(dev)  # row-vector convention, like gaussiansplatting Camera
    center = torch.inverse(Wr)[3, :3]
    return SimpleNamespace(world_view_transform=Wr, FoVx=FOV, FoVy=FOV, camera_center=center, R=R, t=t)


def texture(x, y):
    return torch.stack(
        [0.5 + 0.5 * torch.sin(5 * x), 0.5 + 0.5 * torch.cos(4 * y), 0.5 + 0.5 * torch.sin(3 * (x + y))], dim=-1
    )


def render_plane(cam):
    """Ray-cast the plane Z=0 (world) from `cam`; returns image [H,W,3], view-space depth [H,W]."""
    tan = math.tan(FOV / 2)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    ndc_x = (xs.float() + 0.5) / W * 2 - 1
    ndc_y = (ys.float() + 0.5) / H * 2 - 1
    d_view = torch.stack([ndc_x * tan, ndc_y * tan, torch.ones_like(ndc_x)], dim=-1)
    R, t = cam.R.to(dev), cam.t.to(dev)
    origin = -(R.T @ t)
    d_world = d_view @ R  # R^T d for row vectors
    s = -origin[2] / d_world[..., 2]
    p = origin + s[..., None] * d_world
    depth = (p @ R.T + t)[..., 2]
    return texture(p[..., 0], p[..., 1]), depth


cams = [make_cam(a) for a in (-12, 0, 12)]
imgs, depths = zip(*[render_plane(c) for c in cams])
imgs, depths = list(imgs), list(depths)

# 1. geometry: warp view 1 -> view 0 reproduces view 0 where visible
warped, valid = warp_to_view(depths[0], cams[0], cams[1], imgs[1], depths[1])
err = (warped - imgs[0]).abs().mean(-1)[valid]
print(f"warp error on plane: mean {err.mean():.4f}, valid fraction {valid.float().mean():.2f}")
assert valid.float().mean() > 0.5 and err.mean() < 0.03, "warp geometry/convention is wrong"

# 2. a consistent 'edit' (same colour transform in every view) -> ~no inconsistency
edit_ok = [(0.5 * im + torch.tensor([0.3, 0.1, 0.0], device=dev)).clamp(0, 1) for im in imgs]
inc = edit_inconsistency(cams, depths, imgs, edit_ok, k=2)
m = torch.nan_to_num(inc[1], nan=0.0)
print(f"consistent edit: mean inconsistency {m.mean():.4f}")

# 3. a view-specific blob painted into view 1 only -> inconsistency high at the blob, weight low
edit_bad = [e.clone() for e in edit_ok]
edit_bad[1][40:70, 40:70] = torch.tensor([1.0, 0.0, 1.0], device=dev)
inc_bad = edit_inconsistency(cams, depths, imgs, edit_bad, k=2)
w = inconsistency_to_weight(inc_bad[1], sigma=0.1, min_weight=0.1)
inside, outside = w[40:70, 40:70].mean().item(), w[90:120, 90:120].mean().item()
print(f"weight inside inconsistent blob {inside:.3f}, weight in consistent area {outside:.3f}")
assert inside < 0.5 and outside > 0.9 and m.mean() < 0.02

# 4. misregistration: depth of view 0 is 4% off, so the warp lands a few pixels away from the truth.  A contrast-
#    boosting edit that is perfectly consistent across views must NOT be flagged once we tolerate small misalignment.
edit_boost = [((im - 0.5) * 1.5 + 0.5).clamp(0, 1) for im in imgs]
bad_depths = [depths[0] * 1.04] + depths[1:]
inc_strict = edit_inconsistency(cams, bad_depths, imgs, edit_boost, k=2, tol_px=0)
inc_tol = edit_inconsistency(cams, bad_depths, imgs, edit_boost, k=2, tol_px=2)
s, t = torch.nan_to_num(inc_strict[0]).mean().item(), torch.nan_to_num(inc_tol[0]).mean().item()
print(f"misregistered, consistent contrast-boost edit: strict {s:.4f} vs tolerant {t:.4f}")
assert t < 0.5 * s and t < 0.02, "tolerance should remove most misregistration-induced inconsistency"
print("CONSISTENCY_TEST_OK")

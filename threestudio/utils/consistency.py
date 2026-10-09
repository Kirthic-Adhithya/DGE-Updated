"""Cross-view consistency of edited images, used to make the Gaussian fit robust to contradictory edits.

DGE edits a set of rendered views with a 2D editor and then fits the Gaussians to the edits with equal trust in every
pixel. Even with DGE's attention sharing, some regions of some views still disagree with the same region in neighbouring
views. Fitting those regions makes the 3D model blurry / blotchy.

Idea (ours): measure, for every pixel of every edited view, how much the *edit itself* disagrees with the edits of the
neighbouring views, and trust that pixel less when it does.

  1. Unproject the pixel with the rendered depth, reproject it into a neighbouring camera, and compare the colours.
  2. A plain colour difference is polluted by geometry error and view-dependent effects that exist even without any edit,
     so we subtract the same difference measured on the ORIGINAL (unedited) renders:
         inconsistency = relu( |E_i - warp(E_j)| - |O_i - warp(O_j)| )
     which isolates the disagreement introduced by the editor.
  3. A pixel is consistent if it agrees with at least one visible neighbour (min over neighbours); occluded / out-of-frame
     pixels carry no evidence and keep full weight.
  4. weight = exp(-inconsistency / sigma), floored at `min_weight`.

Conventions follow gaussiansplatting's Camera: row-vector matrices, p_view = p_world_h @ world_view_transform, and a
centred principal point so that ndc = view_xy / (z * tan(fov / 2)).
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _tans(cam) -> Tuple[float, float]:
    return math.tan(cam.FoVx / 2), math.tan(cam.FoVy / 2)


@torch.no_grad()
def warp_to_view(
    depth_i: torch.Tensor,  # [H, W] view-space z of view i (opacity-normalised)
    cam_i,
    cam_j,
    img_j: torch.Tensor,  # [H, W, C] image of view j
    depth_j: Optional[torch.Tensor] = None,  # [H, W] view-space z of view j, for the visibility test
    depth_tol: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample `img_j` at the location where each pixel of view i reprojects. Returns (warped [H,W,C], valid [H,W])."""
    H, W = depth_i.shape
    dev = depth_i.device
    tx, ty = _tans(cam_i)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    ndc_x = (xs.float() + 0.5) / W * 2 - 1
    ndc_y = (ys.float() + 0.5) / H * 2 - 1
    z = depth_i
    p_view = torch.stack([ndc_x * z * tx, ndc_y * z * ty, z, torch.ones_like(z)], dim=-1)  # [H, W, 4]
    p_world = p_view @ torch.inverse(cam_i.world_view_transform.float())
    p_j = p_world @ cam_j.world_view_transform.float()
    zj = p_j[..., 2]
    txj, tyj = _tans(cam_j)
    zc = zj.clamp(min=1e-6)
    u = p_j[..., 0] / (zc * txj)
    v = p_j[..., 1] / (zc * tyj)
    grid = torch.stack([u, v], dim=-1)[None]
    valid = (u.abs() <= 1) & (v.abs() <= 1) & (zj > 1e-3) & (z > 1e-3)
    warped = F.grid_sample(
        img_j.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border", align_corners=False
    )[0].permute(1, 2, 0)
    if depth_j is not None:
        dj = F.grid_sample(depth_j[None, None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
        valid = valid & ((zj - dj).abs() < depth_tol * zj)
    return warped, valid


def tolerant_abs_diff(a: torch.Tensor, b: torch.Tensor, tol_px: int) -> torch.Tensor:
    """mean-over-channels |a - b| minimised over +-tol_px shifts of b.  a, b: [H, W, C].

    The rendered depth is only approximate, so a warped pixel can land 1-2 px away from where it should. At an edge
    that looks like a large colour difference. Taking the best match inside a small window removes this misregistration
    while a genuine disagreement (different colour / content) still survives."""
    if tol_px <= 0:
        return (a - b).abs().mean(-1)
    H, W, _ = a.shape
    bp = F.pad(b.permute(2, 0, 1)[None], (tol_px,) * 4, mode="replicate")[0].permute(1, 2, 0)
    best = None
    for dy in range(2 * tol_px + 1):
        for dx in range(2 * tol_px + 1):
            d = (a - bp[dy : dy + H, dx : dx + W]).abs().mean(-1)
            best = d if best is None else torch.minimum(best, d)
    return best


@torch.no_grad()
def edit_inconsistency(
    cams: List,
    depths: List[torch.Tensor],  # each [H, W]
    origs: List[torch.Tensor],  # each [H, W, 3] unedited renders
    edits: List[torch.Tensor],  # each [H, W, 3] edited images
    k: int = 3,
    depth_tol: float = 0.05,
    blur: int = 7,
    tol_px: int = 2,
) -> List[torch.Tensor]:
    """Per-view [H, W] edit-induced inconsistency; NaN where no neighbour can see the pixel."""
    n = len(cams)
    centers = torch.stack([c.camera_center.float() for c in cams])
    dist = torch.cdist(centers, centers)
    dist.fill_diagonal_(float("inf"))
    nbrs = dist.topk(min(k, n - 1), largest=False).indices.tolist()

    out = []
    for i in range(n):
        best = None
        for j in nbrs[i]:
            both_j = torch.cat([origs[j], edits[j]], dim=-1)  # warp both images with one grid
            warped, valid = warp_to_view(depths[i], cams[i], cams[j], both_j, depths[j], depth_tol)
            e_orig = tolerant_abs_diff(origs[i], warped[..., :3], tol_px)
            e_edit = tolerant_abs_diff(edits[i], warped[..., 3:], tol_px)
            e = F.relu(e_edit - e_orig)
            if blur > 1:  # tolerate small misalignment: compare local averages, not single pixels
                e = F.avg_pool2d(e[None, None], blur, stride=1, padding=blur // 2)[0, 0]
            e = torch.where(valid, e, torch.full_like(e, float("nan")))
            best = e if best is None else torch.fmin(best, e)  # fmin ignores NaN: agree with ANY visible neighbour
        out.append(best)
    return out


def inconsistency_to_weight(inc: torch.Tensor, sigma: float = 0.1, min_weight: float = 0.1) -> torch.Tensor:
    """[H, W] inconsistency -> [H, W] weight in [min_weight, 1]; NaN (no evidence) -> 1."""
    w = torch.exp(-inc / sigma).clamp(min=min_weight)
    return torch.where(torch.isnan(inc), torch.ones_like(w), w)

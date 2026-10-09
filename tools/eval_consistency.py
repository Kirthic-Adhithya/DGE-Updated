"""Evaluate multi-view consistency and edit strength of edited 3DGS models against the original model.

Metrics (all on renders of the final 3D models, from camera views spread over the whole capture):
  inconsistency  mean over pixels of relu(|E_i - warp(E_j)| - |O_i - warp(O_j)|), using depth from the ORIGINAL model
                 (same definition as the training-time weights, but measured on the final 3D result; lower is better)
  frac_bad       fraction of pixels whose inconsistency exceeds 0.05
  edit_strength  mean |E_i - O_i|  (guards against 'improving' consistency by just weakening the edit)

Usage: python tools/eval_consistency.py --source data/tandt/truck --orig outputs/truck_lean/point_cloud/iteration_15000/point_cloud.ply \
           --edited baseline=path/to/last.ply ours=path/to/last.ply --views 48 --out outputs/eval.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel, Scene
from threestudio.utils.consistency import edit_inconsistency

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--orig", required=True)
p.add_argument("--edited", nargs="+", required=True, help="name=path/to/last.ply")
p.add_argument("--views", type=int, default=48)
p.add_argument("--out", default="eval.json")
p.add_argument("--save_dir", default=None, help="optional: write inconsistency heat-maps here")
args = p.parse_args()

tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_eval_tmp")
os.makedirs(tmp, exist_ok=True)
model_args = SimpleNamespace(
    source_path=os.path.abspath(args.source), model_path=tmp, images="images", resolution=2,
    white_background=False, data_device="cpu", eval=False, sh_degree=3,
)
dummy = GaussianModel(sh_degree=0, anchor_weight_init_g0=1.0, anchor_weight_init=0.1, anchor_weight_multiplier=2)
scene = Scene(model_args, dummy, shuffle=False)
all_cams = scene.getTrainCameras()
sel = np.linspace(0, len(all_cams) - 1, args.views).round().astype(int)
cams = [all_cams[i] for i in sel]
pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
bg = torch.zeros(3, device="cuda")


def load(path):
    g = GaussianModel(sh_degree=0, anchor_weight_init_g0=1.0, anchor_weight_init=0.1, anchor_weight_multiplier=2)
    g.load_ply(path)
    return g


@torch.no_grad()
def render_all(g, with_depth=False):
    imgs, depths = [], []
    for c in cams:
        pkg = render(c, g, pipe, bg)
        imgs.append(pkg["render"].clamp(0, 1).permute(1, 2, 0))
        if with_depth:
            alpha = render(c, g, pipe, bg, override_color=torch.ones_like(g.get_xyz))["render"][0]
            d = pkg["depth_3dgs"][0]
            depths.append(torch.where(alpha > 0.5, d / alpha.clamp(min=1e-3), torch.zeros_like(d)))
    return imgs, depths


orig = load(args.orig)
O, D = render_all(orig, with_depth=True)
print(f"{len(cams)} views, {O[0].shape[0]}x{O[0].shape[1]}")
results = {}
for item in args.edited:
    name, path = item.split("=", 1)
    E, _ = render_all(load(path))
    inc = edit_inconsistency(cams, D, O, E, k=3, tol_px=2)
    vals = torch.cat([i[~torch.isnan(i)] for i in inc])
    strict = edit_inconsistency(cams, D, O, E, k=3, tol_px=0)  # no misalignment tolerance
    strict_vals = torch.cat([i[~torch.isnan(i)] for i in strict])
    strength = torch.stack([(e - o).abs().mean() for e, o in zip(E, O)]).mean().item()
    results[name] = {
        "inconsistency_strict_mean": strict_vals.mean().item(),
        "inconsistency_mean": vals.mean().item(),
        "inconsistency_p90": vals.quantile(0.9).item() if vals.numel() < 16_000_000 else float("nan"),
        "frac_bad_gt_0.05": (vals > 0.05).float().mean().item(),
        "edit_strength": strength,
        "pixels_evaluated": int(vals.numel()),
    }
    print(name, json.dumps(results[name], indent=1))
    if args.save_dir:
        import cv2

        os.makedirs(args.save_dir, exist_ok=True)
        rows = []
        for n in range(0, len(cams), max(1, len(cams) // 4)):
            heat = torch.nan_to_num(inc[n], nan=0.0).clamp(0, 0.2) / 0.2
            heat = cv2.applyColorMap((heat.cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            img = cv2.cvtColor((E[n].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            rows.append(np.concatenate([img, heat], axis=0))
        cv2.imwrite(os.path.join(args.save_dir, f"inconsistency_{name}.png"), np.concatenate(rows, axis=1))

with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print("saved", args.out)

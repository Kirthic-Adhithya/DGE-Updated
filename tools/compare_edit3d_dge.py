"""Compare DGE and Edit3D on the same cameras, using Edit3D's saved ADSS masks.

Renders three Gaussian models (original / DGE edit / Edit3D edit) from the 512x512 training cameras that Edit3D saved a
mask for, then reports, per method:
  leak_L1        mean |edit - original| on pixels that should NOT change (mask 'retain' > 0.9; lower is better)
  leak_frac      fraction of those pixels that changed by more than 0.1
  edit_L1        mean |edit - original| inside the edit region (mask 'retain' < 0.1; higher = stronger edit)
  red_gain_edit  mean increase of redness (R - (G+B)/2) in the edit region     (instruction asks for a RED truck)
  red_gain_leak  the same on the retained region (redness leaking where it should not)

IMPORTANT: Edit3D's mask only covers what the query matched (here the cab), so a method that also edits the rest of the
truck (e.g. the wooden bed) is counted as 'leaking' on those pixels. Read leak numbers together with the images.

Usage: python tools/compare_edit3d_dge.py <source> <orig.ply> <dge.ply> <edit3d.ply> <masks_dir> <out.png> <out.json> [n_sheet_views]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.camera_scene import CamScene

source, orig_ply, dge_ply, e3d_ply, masks_dir, out_png, out_json = sys.argv[1:8]
n_sheet = int(sys.argv[8]) if len(sys.argv) > 8 else 4

ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(masks_dir) if f.endswith((".jpg", ".png")))
scene = CamScene(source, h=512, w=512)
pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
bg = torch.zeros(3, device="cuda")


def load(path):
    g = GaussianModel(sh_degree=0, anchor_weight_init_g0=1.0, anchor_weight_init=0.1, anchor_weight_multiplier=2)
    g.load_ply(path)
    return g


@torch.no_grad()
def render_views(g):
    return {i: render(scene.cameras[i], g, pipe, bg)["render"].clamp(0, 1).permute(1, 2, 0) for i in ids}


def load_mask(i):
    m = Image.open(os.path.join(masks_dir, f"{i:05d}.jpg")).convert("L").resize((512, 512))
    return torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0).cuda()


O = render_views(load(orig_ply))
R = {"DGE": render_views(load(dge_ply)), "Edit3D": render_views(load(e3d_ply))}
M = {i: load_mask(i) for i in ids}
print(f"{len(ids)} views; mean 'retain' fraction of the mask = {np.mean([M[i].mean().item() for i in ids]):.3f} "
      "(should be well above 0.5 if white = pixels to keep)")


def red(x):
    return x[..., 0] - 0.5 * (x[..., 1] + x[..., 2])


metrics = {}
for name, X in R.items():
    keep_d, edit_d, keep_n, edit_n, keep_red, edit_red, keep_chg = [], [], 0, 0, [], [], []
    for i in ids:
        diff = (X[i] - O[i]).abs().mean(-1)
        dred = red(X[i]) - red(O[i])
        keep, edit = M[i] > 0.9, M[i] < 0.1
        if keep.any():
            keep_d.append(diff[keep].mean().item()); keep_red.append(dred[keep].mean().item())
            keep_chg.append((diff[keep] > 0.1).float().mean().item())
        if edit.any():
            edit_d.append(diff[edit].mean().item()); edit_red.append(dred[edit].mean().item())
    metrics[name] = {
        "leak_L1": float(np.mean(keep_d)), "leak_frac": float(np.mean(keep_chg)),
        "edit_L1": float(np.mean(edit_d)), "red_gain_edit": float(np.mean(edit_red)),
        "red_gain_leak": float(np.mean(keep_red)), "views": len(ids),
    }
    print(name, json.dumps(metrics[name], indent=1))
json.dump(metrics, open(out_json, "w"), indent=1)

# sheet: original | DGE | Edit3D | retain mask, a few evenly spaced views
pick = [ids[j] for j in np.linspace(0, len(ids) - 1, n_sheet).round().astype(int)]
tile = 384
rows = []
for i in pick:
    ims = [O[i], R["DGE"][i], R["Edit3D"][i], M[i][..., None].repeat(1, 1, 3)]
    row = np.concatenate([(im.cpu().numpy() * 255).astype(np.uint8) for im in ims], axis=1)
    rows.append(Image.fromarray(row).resize((4 * tile, tile)))
sheet = Image.new("RGB", (4 * tile, 24 + n_sheet * tile), "white")
d = ImageDraw.Draw(sheet)
for k, label in enumerate(("original", "DGE", "Edit3D", "Edit3D mask (white = keep)")):
    d.text((k * tile + 8, 6), label, fill="black")
for r, row in enumerate(rows):
    sheet.paste(row, (0, 24 + r * tile))
sheet.save(out_png)
print("saved", out_png)

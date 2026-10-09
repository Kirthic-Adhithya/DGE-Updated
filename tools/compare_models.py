"""Compare any number of edited Gaussian models against the original on the SAME cameras, using an ADSS mask folder.

Per model it reports (mask 'retain' = white = pixels that should not change; edit region = black):
  leak_L1        mean |edit - original| on retained pixels (mask > 0.9)      lower = better preservation
  leak_frac      share of retained pixels changed by more than 0.1            lower = better
  edit_L1        mean |edit - original| inside the edit region (mask < 0.1)   higher = stronger edit
  red_abs_edit   mean redness (R - (G+B)/2) of the edit region AFTER editing  (the original's value is reported too)
  red_gain_edit  red_abs_edit minus the original's: positive = became redder
  red_gain_leak  the same on the retained region: redness that leaked where it should not

Read these together with the images: a mask only covers what its query matched, so a method that edits MORE of the truck
than the mask covers is counted as leaking on those pixels.

Usage: python tools/compare_models.py <source> <orig.ply> <masks_dir> <out.png> <out.json> <n_sheet_views> name=model.ply [name=model.ply ...]
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

source, orig_ply, masks_dir, out_png, out_json, n_sheet = sys.argv[1:7]
n_sheet = int(n_sheet)
models = dict(item.split("=", 1) for item in sys.argv[7:])

ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(masks_dir) if f.endswith((".jpg", ".png")))
scene = CamScene(source, h=512, w=512)
pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
bg = torch.zeros(3, device="cuda")


def load(path):
    g = GaussianModel(sh_degree=0, anchor_weight_init_g0=1.0, anchor_weight_init=0.1, anchor_weight_multiplier=2)
    g.load_ply(path)
    return g


@torch.no_grad()
def render_views(path):
    g = load(path)
    out = {i: render(scene.cameras[i], g, pipe, bg)["render"].clamp(0, 1).permute(1, 2, 0) for i in ids}
    del g
    torch.cuda.empty_cache()
    return out


def load_mask(i):
    m = Image.open(os.path.join(masks_dir, f"{i:05d}.jpg")).convert("L").resize((512, 512))
    return torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0).cuda()


def red(x):
    return x[..., 0] - 0.5 * (x[..., 1] + x[..., 2])


O = render_views(orig_ply)
M = {i: load_mask(i) for i in ids}
print(f"{len(ids)} views; mean retain fraction of the mask = {np.mean([M[i].mean().item() for i in ids]):.3f}")
orig_red_edit = float(np.mean([red(O[i])[M[i] < 0.1].mean().item() for i in ids if (M[i] < 0.1).any()]))
R, metrics = {}, {"_original": {"red_abs_edit": orig_red_edit, "views": len(ids)}}
for name, path in models.items():
    X = R[name] = render_views(path)
    keep_d, keep_chg, keep_red, edit_d, edit_red = [], [], [], [], []
    for i in ids:
        diff = (X[i] - O[i]).abs().mean(-1)
        keep, edit = M[i] > 0.9, M[i] < 0.1
        if keep.any():
            keep_d.append(diff[keep].mean().item())
            keep_chg.append((diff[keep] > 0.1).float().mean().item())
            keep_red.append((red(X[i]) - red(O[i]))[keep].mean().item())
        if edit.any():
            edit_d.append(diff[edit].mean().item())
            edit_red.append(red(X[i])[edit].mean().item())
    metrics[name] = {
        "leak_L1": float(np.mean(keep_d)), "leak_frac": float(np.mean(keep_chg)),
        "edit_L1": float(np.mean(edit_d)), "red_abs_edit": float(np.mean(edit_red)),
        "red_gain_edit": float(np.mean(edit_red)) - orig_red_edit, "red_gain_leak": float(np.mean(keep_red)),
        "views": len(ids),
    }
    print(name, json.dumps(metrics[name], indent=1))
json.dump(metrics, open(out_json, "w"), indent=1)

# sheet: original | each model | retain mask, a few evenly spaced views
pick = [ids[j] for j in np.linspace(0, len(ids) - 1, n_sheet).round().astype(int)]
cols = ["original"] + list(models) + ["mask (white = keep)"]
tile = 320
sheet = Image.new("RGB", (len(cols) * tile, 24 + n_sheet * tile), "white")
d = ImageDraw.Draw(sheet)
for k, label in enumerate(cols):
    d.text((k * tile + 8, 6), label, fill="black")
for r, i in enumerate(pick):
    ims = [O[i]] + [R[n][i] for n in models] + [M[i][..., None].repeat(1, 1, 3)]
    for k, im in enumerate(ims):
        arr = (im.cpu().numpy() * 255).astype(np.uint8)
        sheet.paste(Image.fromarray(arr).resize((tile, tile)), (k * tile, 24 + r * tile))
sheet.save(out_png)
print("saved", out_png)

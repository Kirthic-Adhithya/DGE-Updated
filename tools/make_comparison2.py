"""Side-by-side sheet: original render | baseline DGE | ours (consistency-weighted), final 3D renders, plus crops.

Usage: python tools/make_comparison2.py <baseline_save_dir> <ours_save_dir> <out.png> [n_views]
Uses the validation renders saved during each run (it1000-val = [original | 2D edit], render_it1000-val = final 3D render).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import os
import sys

from PIL import Image, ImageDraw

base, ours, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 4


def panels(save, name):
    pair = Image.open(os.path.join(save, "it1000-val", name)).convert("RGB")
    h, w = pair.height, pair.width // 2
    orig = pair.crop((0, 0, w, h))
    final = Image.open(os.path.join(save, "render_it1000-val", name)).convert("RGB")
    final = final.crop((0, 0, min(final.width, final.height), final.height)).resize((w, h))
    return orig, final


common = sorted(
    set(os.listdir(os.path.join(base, "it1000-val"))) & set(os.listdir(os.path.join(ours, "it1000-val"))),
    key=lambda s: int(s.split(".")[0]),
)[:n]
rows = []
for name in common:
    orig, fb = panels(base, name)
    _, fo = panels(ours, name)
    w, h = orig.size
    row = Image.new("RGB", (3 * w, h))
    for i, im in enumerate((orig, fb, fo)):
        row.paste(im, (i * w, 0))
    rows.append(row)

pad = 28
sheet = Image.new("RGB", (rows[0].width, pad + sum(r.height for r in rows)), "white")
d = ImageDraw.Draw(sheet)
w = rows[0].width // 3
for i, label in enumerate(("original", "DGE baseline", "ours (consistency-weighted)")):
    d.text((i * w + 8, 8), label, fill="black")
y = pad
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height
sheet.save(out)
print("saved", out, sheet.size)

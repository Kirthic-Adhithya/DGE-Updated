"""Build a before/after sheet from a DGE run's save/ folder.

Rows = validation views. Columns = original render | 2D edit used as target | final edited-3DGS render.
Usage: python tools/make_comparison.py <run_save_dir> <out.png> [n_views]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import os
import sys

from PIL import Image

save, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
names = sorted(os.listdir(os.path.join(save, "it1000-val")), key=lambda s: int(s.split(".")[0]))[:n]

rows = []
for name in names:
    pair = Image.open(os.path.join(save, "it1000-val", name)).convert("RGB")  # [original | 2D edit]
    h = pair.height
    w = pair.width // 2
    orig, edit = pair.crop((0, 0, w, h)), pair.crop((w, 0, 2 * w, h))
    final = Image.open(os.path.join(save, "render_it1000-val", name)).convert("RGB")
    final = final.crop((0, 0, final.height, final.height)) if final.width > final.height else final  # first panel = RGB render
    final = final.resize((w, h))
    row = Image.new("RGB", (3 * w, h))
    for i, im in enumerate((orig, edit, final)):
        row.paste(im, (i * w, 0))
    rows.append(row)

sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
y = 0
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height
sheet.save(out)
print("saved", out, sheet.size)

"""Trace the fold centre-lines of a reference image from its depth map.

  <python with scipy, skimage> trace_folds.py <depth.npy> <alpha.npy> <out folds.json> [<preview.png> <reference.jpg>]

A fold of a rolled, tube-like form is a ridge in the depth map (nearest along its centre). For every column the
local depth maxima along the vertical are ridge points; points in neighbouring columns within a few pixels are
linked into one polyline; short or faint chains are dropped. Each fold is written as image coordinates (0..1),
depth (0..1, larger = nearer) and the local half-width of the ridge (where depth falls by half its prominence), which
sets the tube radius. image_to_3d.py sweeps a round tube along each one.
"""
import sys, json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

dp, ap_, out = sys.argv[1], sys.argv[2], sys.argv[3]
D = gaussian_filter(np.load(dp).astype(np.float64), 4)
A = np.load(ap_)
H, W = D.shape
STEP = 6
chains = []
open_ = []
for x in range(0, W, STEP):
    col = np.where(A[:, x] > 0.35, D[:, x], 0)
    peaks, prop = find_peaks(col, prominence=0.035, distance=40, width=(8, None), rel_height=0.5)
    pts = [(x, int(p), float(col[p]), float(w) / 2) for p, w in zip(peaks, prop["widths"])]
    nxt = []
    used = set()
    for ch in open_:
        lx, ly = ch[-1][0], ch[-1][1]
        best = min(((abs(p[1] - ly), i) for i, p in enumerate(pts) if i not in used), default=(1e9, -1))
        if best[0] < 26:
            ch.append(pts[best[1]])
            used.add(best[1])
            nxt.append(ch)
        else:
            chains.append(ch)
    for i, p in enumerate(pts):
        if i not in used:
            nxt.append([p])
    open_ = nxt
chains += open_
folds = []
for ch in chains:
    if len(ch) * STEP < W * 0.12:
        continue
    arr = np.array(ch, dtype=np.float64)
    folds.append({"points": [[round(x / W, 4), round(y / H, 4), round(d, 4), round(r / H, 4)] for x, y, d, r in arr]})
folds.sort(key=lambda f: np.mean([p[1] for p in f["points"]]))
json.dump({"size": [W, H], "folds": folds}, open(out, "w"), indent=1)
print(json.dumps({"folds": len(folds), "points": [len(f["points"]) for f in folds]}))
if len(sys.argv) > 5:
    from PIL import Image, ImageDraw
    im = Image.open(sys.argv[5]).convert("RGB")
    dr = ImageDraw.Draw(im)
    for k, f in enumerate(folds):
        xy = [(p[0] * W, p[1] * H) for p in f["points"]]
        dr.line(xy, fill=(255, 40, 40), width=6)
        dr.text(xy[0], str(k), fill=(255, 255, 255))
    im.resize((W // 2, H // 2)).save(sys.argv[4])

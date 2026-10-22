"""Place a generated mesh on the reference's silhouette without rendering.

  <python with trimesh> place_mesh.py <preset.json> [--write]

Projects the mesh's vertices through the same camera and box fit as image_to_3d.py, splats them into a coarse
silhouette and grid-searches mesh.scale, mesh.offset and mesh.yaw for the best overlap (IoU) with the image's soft
silhouette. Seconds instead of renders; --write stores the winner in the preset.
"""
import sys, json, math, itertools
from pathlib import Path
import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_fill_holes

sp = Path(sys.argv[1]).resolve()
S = json.loads(sp.read_text(encoding="utf-8"))
rel = lambda p: (sp.parent / p).resolve()
MB, G = S["mesh"], S["image"]
W, H = G["width"], G["height"]
TAN_V = (24 / 2) / S["camera"]["lens"]
TAN_H = TAN_V * W / H
m = trimesh.load(rel(MB["path"]), force="mesh")
V = np.asarray(m.vertices)
V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], -1)  # glTF y-up to Blender z-up, as the importer does
al = np.load(rel(G["mask"]))
GW, GH = 150, 268
ys, xs = np.where(al > 0.5)
u0, u1 = xs.min() / al.shape[1] * 2 - 1, xs.max() / al.shape[1] * 2 - 1
v1, v0 = 1 - ys.min() / al.shape[0] * 2, 1 - ys.max() / al.shape[0] * 2
tgt = al[np.linspace(0, al.shape[0] - 1, GH).astype(int)][:, np.linspace(0, al.shape[1] - 1, GW).astype(int)] > 0.5

def place(scale, ox, oy, yaw, dist):
    a = math.radians(yaw)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    P = V @ R.T
    lo, hi = V.min(0), V.max(0)
    sx = (u1 - u0) * TAN_H * dist / (hi[0] - lo[0])
    sz = (v1 - v0) * TAN_V * dist / (hi[2] - lo[2])
    s_ = (sx + sz) / 2 * scale
    ctr = (lo + hi) / 2
    loc = np.array([((u0 + u1) / 2 + 2 * ox) * TAN_H * dist - ctr[0] * s_, dist - ctr[1] * s_, ((v0 + v1) / 2 + 2 * oy) * TAN_V * dist - ctr[2] * s_])
    X = P * s_ + loc
    u = X[:, 0] / (X[:, 1] * TAN_H)
    v = X[:, 2] / (X[:, 1] * TAN_V)
    c = ((u + 1) / 2 * GW).astype(int)
    r = ((1 - v) / 2 * GH).astype(int)
    ok = (c >= 0) & (c < GW) & (r >= 0) & (r < GH)
    img = np.zeros((GH, GW), bool)
    img[r[ok], c[ok]] = True
    img = binary_fill_holes(binary_closing(img, iterations=2))
    return float((img & tgt).sum() / max(1, (img | tgt).sum()))

dist = MB["depth"]
best = (place(MB.get("scale", 1), *MB.get("offset", [0, 0]), MB.get("yaw", 0), dist), MB.get("scale", 1), *MB.get("offset", [0, 0]), MB.get("yaw", 0))
print("start IoU", round(best[0], 3))
for sc, ox, oy, yw in itertools.product(np.linspace(0.8, 1.3, 11), np.linspace(-0.15, 0.15, 7), np.linspace(-0.15, 0.15, 7), (-30, -20, -10, 0, 10, 20, 30)):
    s = place(sc, ox, oy, yw, dist)
    if s > best[0]:
        best = (s, sc, ox, oy, yw)
# refine around the winner
for sc, ox, oy, yw in itertools.product(best[1] + np.linspace(-0.04, 0.04, 5), best[2] + np.linspace(-0.02, 0.02, 5), best[3] + np.linspace(-0.02, 0.02, 5), best[4] + np.linspace(-5, 5, 5)):
    s = place(sc, ox, oy, yw, dist)
    if s > best[0]:
        best = (s, sc, ox, oy, yw)
res = {"iou": round(best[0], 3), "scale": round(float(best[1]), 3), "offset": [round(float(best[2]), 3), round(float(best[3]), 3)], "yaw": round(float(best[4]), 1)}
print("RESULT", json.dumps(res))
if "--write" in sys.argv:
    S["mesh"].update({"scale": res["scale"], "offset": res["offset"], "yaw": res["yaw"]})
    sp.write_text(json.dumps(S, indent=1), encoding="utf-8")

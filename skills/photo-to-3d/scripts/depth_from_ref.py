"""Depth map of the reference photo, so the 3D shape comes from the photo instead of being hand-built.

  python depth_from_ref.py <reference.jpg> <out dir>

Model: Depth Anything V2 Small (Apache 2.0; the Base and Large weights are non-commercial, so they are not used).
Writes depth.npy (float32, relative inverse depth, larger = nearer, at the reference's size), depth.png (preview)
and mask.npy (True where the forms are, False on the background, from colour distance to the colour in the top-left and
bottom-left corners; the photo needs a fairly plain background there). Uses an NVIDIA card when torch can see one, else the
processor. Needs torch and transformers.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
    sys.exit("usage: python depth_from_ref.py <reference image> <out folder>")
ref, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
try:
    img = Image.open(ref).convert("RGB")
except Exception as e:
    sys.exit(f"cannot read image {ref}: {e}")
# the model's default input is 518 px tall, which steps the surface; run it at 1036 px (a multiple of its 14 px patch)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
mid = "depth-anything/Depth-Anything-V2-Small-hf"
proc = AutoImageProcessor.from_pretrained(mid)
model = AutoModelForDepthEstimation.from_pretrained(mid).to("cuda" if torch.cuda.is_available() else "cpu").eval()
inp = proc(images=img, return_tensors="pt", size={"height": 1036, "width": 588}, do_resize=True, keep_aspect_ratio=False)
with torch.no_grad():
    d = model(**{k: v.to(model.device) for k, v in inp.items()}).predicted_depth.squeeze().float().cpu().numpy()
d = np.asarray(Image.fromarray(d).resize(img.size, Image.BICUBIC), dtype=np.float32)
d = (d - d.min()) / (d.max() - d.min())
np.save(out / "depth.npy", d)
Image.fromarray((d * 255).astype(np.uint8)).save(out / "depth.png")

px = np.asarray(img, dtype=np.float32)
bg = np.median(np.concatenate([px[:40, :40].reshape(-1, 3), px[-40:, :40].reshape(-1, 3)]), 0)
dist = np.linalg.norm(px - bg, axis=-1)
mask = dist > 38
np.save(out / "mask.npy", mask)
Image.fromarray((mask * 255).astype(np.uint8)).save(out / "mask.png")
print("RESULT " + json.dumps({"device": "cuda" if torch.cuda.is_available() else "cpu", "size": list(img.size),
                              "background": [round(float(v)) for v in bg], "form_share": round(float(mask.mean()), 3)}))

"""Closed 3D mesh of the forms in a reference image, with TripoSG (MIT), on an NVIDIA card.

  <python of the TripoSG install> image_to_mesh.py <reference.jpg> <alpha.npy> <out.glb>
        [--engine DIR] [--steps 50] [--seed 42] [--faces 200000]

--engine (or the TRIPOSG_DIR environment variable) is the folder of a TripoSG checkout whose pretrained_weights/TripoSG
folder holds the downloaded weights; see the skill's SKILL.md. The image is cut out with the soft silhouette in alpha.npy
and passed with transparency, so TripoSG never loads its default background remover (BRIA RMBG-1.4, non-commercial
licence). TripoSG needs an NVIDIA card; with none the script stops with one plain line.
"""
import sys, argparse, os
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("alpha"); ap.add_argument("out")
ap.add_argument("--engine", default=os.environ.get("TRIPOSG_DIR"))
ap.add_argument("--steps", type=int, default=50); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--faces", type=int, default=200000)
a = ap.parse_args()
if not a.engine or not (Path(a.engine) / "triposg").is_dir():
    sys.exit("TripoSG not found: pass --engine <folder of a TripoSG checkout> or set TRIPOSG_DIR (the folder holding triposg/ and scripts/)")
ENGINE = Path(a.engine).resolve()
if not (ENGINE / "pretrained_weights" / "TripoSG").is_dir():
    sys.exit(f"TripoSG weights missing: download VAST-AI/TripoSG into {ENGINE / 'pretrained_weights' / 'TripoSG'} (no other model is needed)")
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "scripts"))
import types
# diso (a compiled GPU add-on, no Windows wheel) serves only the fast mesh extraction; this runner uses the
# standard extraction instead, so an empty stand-in lets the engine import without editing its files
sys.modules.setdefault("diso", types.SimpleNamespace(DiffDMC=None))
import torch
if not torch.cuda.is_available():
    sys.exit("TripoSG needs an NVIDIA card and this Python's torch cannot see one; skip this step (the depth surface works without a mesh)")
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image
from inference_triposg import simplify_mesh
import trimesh

img = np.asarray(Image.open(a.ref).convert("RGB"))
al = (np.clip(np.load(a.alpha), 0, 1) * 255).astype(np.uint8)
rgba = Path(a.out).with_suffix(".input.png")
Image.fromarray(np.dstack([img, al])).save(rgba)
pil = prepare_image(str(rgba), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)
# load straight at half precision: the full-precision copy would need about 7.5 GB of main memory on the way
pipe = TripoSGPipeline.from_pretrained(str(ENGINE / "pretrained_weights" / "TripoSG"), torch_dtype=torch.float16).to("cuda")
# the standard extraction queries the shape decoder with full-precision points, so that one part stays full precision
pipe.vae.to(torch.float32)
_dec = pipe.vae.decode
pipe.vae.decode = lambda z, *a_, **k_: _dec(z.float(), *a_, **k_)
out = pipe(image=pil, generator=torch.Generator(device=pipe.device).manual_seed(a.seed),
           num_inference_steps=a.steps, guidance_scale=7.0, use_flash_decoder=False).samples[0]
mesh = trimesh.Trimesh(out[0].astype(np.float32), np.ascontiguousarray(out[1]))
if a.faces > 0:
    mesh = simplify_mesh(mesh, a.faces)
mesh.export(a.out)
print("RESULT", {"verts": len(mesh.vertices), "faces": len(mesh.faces), "bounds": mesh.bounds.round(3).tolist()})

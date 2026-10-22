"""Surface directions from the reference photo, and a depth map that agrees with them .

  <card python> normals_from_ref.py --image depth/reference.jpg --depth depth/depth.npy --mask depth/alpha.npy
                                    --out depth/ [--res 768] [--weight 0.05] [--iters 1500]

Needs torch, diffusers, psutil and transformers; uses an NVIDIA card when torch can see one (about 3.3 GB of card memory
at 1200 x 2141), else the processor (slow).

1. Normals: Marigold normals v1.1 (prs-eth; code Apache 2.0, checkpoint CreativeML Open RAIL++-M, see LICENCES.md; diffusers) estimate which way each visible surface faces.
   Depth from one photo is soft and relative; normals carry the crisp turn of every fold edge and curve.
2. Refined depth: a least-squares fit, by gradient descent on the card, of a depth map whose slopes follow the
   normals while staying close to the original depth (weight = how hard it holds to the original). The scale
   between normal slopes and depth units is fitted too. Only pixels inside the forms on both sides of a step count,
   so the jump at an outline never pulls the surface.

Writes normals.npy (H x W x 3, camera space: x right, y up, z towards the viewer), normals.png and depth-refined.npy
(same range and meaning as depth.npy: 0..1, larger = nearer). Prints one RESULT line with the card peak memory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

MODEL = "prs-eth/marigold-normals-v1-1"


def estimate_normals(img: Image.Image, res: int, dev: str) -> np.ndarray:
    from diffusers import MarigoldNormalsPipeline
    kw = {"variant": "fp16", "torch_dtype": torch.float16} if dev == "cuda" else {}
    pipe = MarigoldNormalsPipeline.from_pretrained(MODEL, **kw).to(dev)
    out = pipe(img, processing_resolution=res, match_input_resolution=True, ensemble_size=1, output_type="np")
    n = np.asarray(out.prediction[0], dtype=np.float32)
    del pipe
    if dev == "cuda":
        torch.cuda.empty_cache()
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def refine_depth(d0: np.ndarray, n: np.ndarray, mask: np.ndarray, weight: float, iters: int, dev: str) -> tuple[np.ndarray, float]:
    """Depth whose row/column slopes follow the normals, held to d0 by `weight`; returns it and the fitted scale."""
    nz = np.clip(n[..., 2], 0.15, None)
    gx, gy = -n[..., 0] / nz, n[..., 1] / nz  # slope per pixel along columns and down rows, for "nearer = larger"
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
    D0, GX, GY, M = T(d0), T(gx), T(gy), T(mask)
    mx, my = M[:, 1:] * M[:, :-1], M[1:, :] * M[:-1, :]
    # starting scale: regress the original depth's slopes on the normal slopes inside the forms
    ddx, ddy = D0[:, 1:] - D0[:, :-1], D0[1:, :] - D0[:-1, :]
    gxs, gys = GX[:, :-1], GY[:-1, :]
    s0 = float(((ddx * gxs * mx).sum() + (ddy * gys * my).sum()) / ((gxs * gxs * mx).sum() + (gys * gys * my).sum() + 1e-9))
    D = D0.clone().requires_grad_(True)
    S = torch.tensor(max(s0, 1e-5), device=dev).log().requires_grad_(True)
    opt = torch.optim.Adam([D, S], lr=2e-3)
    for _ in range(iters):
        opt.zero_grad()
        s = S.exp()
        ex = (D[:, 1:] - D[:, :-1] - s * gxs) * mx
        ey = (D[1:, :] - D[:-1, :] - s * gys) * my
        loss = (ex.pow(2).sum() + ey.pow(2).sum()) / (mx.sum() + my.sum()) + weight * ((D - D0) * M).pow(2).sum() / M.sum()
        loss.backward()
        opt.step()
    out = D.detach().cpu().numpy()
    out = np.where(mask > 0.5, out, d0)
    return np.clip(out, 0, 1).astype(np.float32), float(S.exp())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--weight", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--reuse", action="store_true", help="skip the model when normals.npy already exists")
    ap.add_argument("--name", default="depth-refined", help="output file stem, so several weights can sit side by side")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    img = Image.open(a.image).convert("RGB")
    npath = out / "normals.npy"
    if a.reuse and npath.is_file():
        n = np.load(npath)
    else:
        n = estimate_normals(img, a.res, dev)
        np.save(npath, n)
        Image.fromarray(((n * 0.5 + 0.5) * 255).round().astype(np.uint8)).save(out / "normals.png")
    d0 = np.load(a.depth).astype(np.float32)
    mask = (np.load(a.mask) > 0.5).astype(np.float32)
    assert n.shape[:2] == d0.shape == mask.shape, (n.shape, d0.shape, mask.shape)
    d1, scale = refine_depth(d0, n, mask, a.weight, a.iters, dev)
    np.save(out / f"{a.name}.npy", d1)
    Image.fromarray((d1 * 255).round().astype(np.uint8)).save(out / f"{a.name}.png")
    peak = round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else 0
    # peak main memory, so the next card booking asks for the real size (Windows reports the peak working set)
    ram = round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20)
    print("RESULT " + json.dumps({"device": dev, "model": MODEL, "peak_mb": peak, "peak_ram_mb": ram, "scale": round(scale, 6),
                                  "mean_change": round(float(np.abs(d1 - d0)[mask > 0].mean()), 4)}))


if __name__ == "__main__":
    main()

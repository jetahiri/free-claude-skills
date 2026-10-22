"""Depth worker for the reference matcher: one model load, a depth map per image, then exit.

  python refmatch_depth.py [--device auto|cuda|cpu] <out dir> <image> [<image> ...]

Run by refmatch.py in the same Python. Needs torch and transformers. --device auto uses an NVIDIA card when torch can see
one and the processor otherwise; --device cuda stops with a plain message when torch cannot see a card.
Model: Depth Anything V2 Small (Apache 2.0; the Base and Large weights are non-commercial, so they are not used).
For each input it writes <stem>.depth.npy (float32 relative inverse depth scaled 0..1, larger = nearer, at the image's
size) and <stem>.depth.png, then prints one RESULT line with the device and the peak card memory.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def pick_device(want: str) -> str:
    has_card = torch.cuda.is_available()
    if want == "cuda" and not has_card:
        sys.exit("--device cuda was asked for, but this Python's torch cannot see an NVIDIA card "
                 "(install the CUDA build of torch, or use --device cpu or --device auto)")
    return "cuda" if want == "cuda" or (want == "auto" and has_card) else "cpu"


def main(out: Path, images: list[Path], want: str) -> None:
    device = pick_device(want)
    from transformers import pipeline
    out.mkdir(parents=True, exist_ok=True)
    pipe = pipeline("depth-estimation", model=MODEL, device=0 if device == "cuda" else -1)
    done = []
    for p in images:
        img = Image.open(p).convert("RGB")
        d = pipe(img)["predicted_depth"].squeeze().float().cpu().numpy()
        d = np.asarray(Image.fromarray(d).resize(img.size, Image.BICUBIC), dtype=np.float32)
        span = float(d.max() - d.min())
        d = (d - d.min()) / span if span > 0 else np.zeros_like(d)
        np.save(out / f"{p.stem}.depth.npy", d)
        Image.fromarray((d * 255).astype(np.uint8)).save(out / f"{p.stem}.depth.png")
        done.append(p.stem)
    peak = round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0
    print("RESULT " + json.dumps({"device": device, "model": MODEL, "images": done, "peak_card_mb": peak}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    main(Path(a.out), [Path(x) for x in a.images], a.device)

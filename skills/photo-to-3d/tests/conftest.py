"""Shared fixtures: a synthetic depth map and mask (a soft blob), so no photo ships with the tests."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="session")
def depth_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("depth")
    h, w = 160, 90
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / 28) ** 2 + ((yy - h / 2) / 45) ** 2)
    depth = np.clip(1 - r, 0, 1).astype(np.float32)
    np.save(d / "depth.npy", depth)
    np.save(d / "mask.npy", r < 1)
    Image.fromarray((np.dstack([depth] * 3) * 255).astype(np.uint8)).save(d / "reference.jpg")
    return d


@pytest.fixture
def small_preset(depth_dir, tmp_path) -> Path:
    """The starter preset shrunk to the synthetic image and a render that takes seconds."""
    p = json.loads((SKILL / "presets" / "starter.json").read_text(encoding="utf-8"))
    p["image"].update({"image": str(depth_dir / "reference.jpg"), "depth": str(depth_dir / "depth.npy"),
                       "mask": str(depth_dir / "mask.npy"), "width": 90, "height": 160})
    p["geometry"]["grid"] = [90, 160]
    p["render"]["samples"] = 4
    out = tmp_path / "preset.json"
    out.write_text(json.dumps(p), encoding="utf-8")
    return out

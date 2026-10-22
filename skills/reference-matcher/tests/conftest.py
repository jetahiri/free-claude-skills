"""Shared fixtures: every test image is generated here from numpy, so no third-party picture ships with the tests."""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import refmatch  # noqa: E402

W, H = 360, 640
WITH_MODELS = os.environ.get("REFMATCH_TEST_MODELS") == "1"
needs_models = pytest.mark.skipif(not WITH_MODELS, reason="set REFMATCH_TEST_MODELS=1 to run tests that load models")


def _scene(blobs, bg_top=(18, 70, 66), bg_bottom=(10, 44, 46), tint=(1.0, 1.0, 1.0)) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (yy / H)[..., None]
    img = (np.array(bg_top) * (1 - t) + np.array(bg_bottom) * t).astype(np.float32)
    for cx, cy, rx, ry, col in blobs:
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        body = np.clip(1 - d, 0, 1)[..., None]
        shade = (0.55 + 0.45 * np.clip(1 - np.sqrt(((xx - cx + rx * 0.3) / rx) ** 2 + ((yy - cy + ry * 0.3) / ry) ** 2), 0, 1))[..., None]
        img = np.where(d[..., None] < 1, np.array(col) * shade * (0.6 + 0.4 * body), img)
    return np.clip(img * np.array(tint), 0, 255).astype(np.uint8)


BLOBS = [(130, 170, 95, 62, (130, 200, 168)), (210, 330, 120, 70, (120, 190, 160)), (150, 500, 100, 66, (140, 205, 175))]
OTHER = [(250, 120, 60, 110, (200, 120, 90)), (110, 420, 140, 40, (190, 110, 80)), (260, 540, 50, 50, (210, 140, 100))]


@pytest.fixture(scope="session")
def fix(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("fixtures")
    ref = Image.fromarray(_scene(BLOBS))
    ref.save(d / "reference.png")
    ref.filter(ImageFilter.GaussianBlur(1.2)).save(d / "close.png")
    Image.fromarray(_scene(BLOBS, tint=(0.8, 0.85, 0.7))).save(d / "dim.png")
    Image.fromarray(_scene(OTHER, bg_top=(60, 40, 30), bg_bottom=(40, 24, 20))).save(d / "wrong.png")
    rgba = np.dstack([np.asarray(ref), np.full((H, W), 255, np.uint8)])
    Image.fromarray(rgba).save(d / "clay-rgba.png")
    return d


@pytest.fixture
def REF(fix) -> Path:
    return fix / "reference.png"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Own records file per test, never the default beside the bars; the perceptual model stays off unless asked for."""
    monkeypatch.setenv("REFMATCH_DB", str(tmp_path / "records.sqlite"))
    if not WITH_MODELS:
        monkeypatch.setattr(refmatch, "lpips_distance", lambda a, b, notes: notes.append("perceptual score off in tests") or None)


@pytest.fixture
def bars(tmp_path, monkeypatch):
    p = tmp_path / "bars.json"
    shutil.copy(SCRIPTS / "refmatch_bars.json", p)
    monkeypatch.setattr(refmatch, "BARS_FILE", p)
    return p

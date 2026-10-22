"""Photo to 3D: the Blender run, the failure cases a viewer can hit, and the small tools that need no model.
Blender tests skip cleanly when Blender is not installed; nothing here downloads a model."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import inverse_fit

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
THREADS = "4"


def _blender():
    try:
        return inverse_fit.blender_exe()
    except SystemExit:
        pytest.skip("Blender is not installed (set BLENDER to blender.exe to run the Blender tests)")


def _run_blender(preset, out, *extra, env=None):
    cmd = [_blender(), "-b", "--factory-startup", "--threads", THREADS, "--python-exit-code", "1", "--python",
           str(SCRIPTS / "image_to_3d.py"), "--", "--settings", str(preset), "--stage", "final", "--out", str(out),
           "--samples", "4", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def test_render_on_the_processor_writes_a_picture(small_preset, tmp_path):
    out = tmp_path / "final.png"
    r = _run_blender(small_preset, out, "--device", "cpu")
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    result = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT "))[7:])
    assert result["device"] == "NONE" and result["stage"] == "final"
    img = np.asarray(Image.open(out))
    assert img.shape[:2] == (160, 90) and img.std() > 1


def test_no_card_is_said_in_one_plain_line(small_preset, tmp_path):
    # hiding the card from the driver reproduces a viewer without an NVIDIA card
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r = _run_blender(small_preset, tmp_path / "f.png", "--device", "auto", env=env)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert "no NVIDIA card found" in r.stderr and '"device": "NONE"' in r.stdout


def test_missing_depth_file_names_the_fix(small_preset, tmp_path):
    p = json.loads(small_preset.read_text())
    p["image"]["depth"] = str(tmp_path / "nope.npy")
    small_preset.write_text(json.dumps(p))
    r = _run_blender(small_preset, tmp_path / "f.png", "--device", "cpu")
    assert r.returncode != 0 and "missing depth file" in (r.stdout + r.stderr) and "depth_from_ref.py" in (r.stdout + r.stderr)


def test_blender_missing_is_a_plain_message(monkeypatch, tmp_path):
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    with pytest.raises(SystemExit, match="Blender not found.*blender.org"):
        inverse_fit.blender_exe()


def test_blender_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("BLENDER", str(tmp_path / "my-blender"))
    assert inverse_fit.blender_exe() == str(tmp_path / "my-blender")


def _py(script, *args, env=None):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], capture_output=True, text=True,
                          timeout=300, env=env)


def test_a_file_that_is_not_an_image_stops_the_depth_step(tmp_path):
    pytest.importorskip("torch")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    r = _py("depth_from_ref.py", bad, tmp_path / "out")
    assert r.returncode != 0 and "cannot read image" in r.stderr and "Traceback" not in r.stderr


def test_mesh_step_without_an_engine_is_a_plain_message(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "TRIPOSG_DIR"}
    r = _py("image_to_mesh.py", tmp_path / "a.jpg", tmp_path / "a.npy", tmp_path / "o.glb", env=env)
    assert r.returncode != 0 and "TripoSG not found" in r.stderr and "--engine" in r.stderr


def test_mesh_step_without_weights_names_the_download(tmp_path):
    (tmp_path / "engine" / "triposg").mkdir(parents=True)
    r = _py("image_to_mesh.py", tmp_path / "a.jpg", tmp_path / "a.npy", tmp_path / "o.glb", "--engine", tmp_path / "engine")
    assert r.returncode != 0 and "VAST-AI/TripoSG" in r.stderr


def test_fit_without_the_matcher_skill_says_where_to_get_it(tmp_path):
    env = dict(os.environ, REFMATCH_DIR=str(tmp_path / "nowhere"))
    r = _py("inverse_fit.py", "score", "--ref", "a", "--render", "b", env=env)
    assert r.returncode != 0 and "reference-matcher skill" in r.stderr


def test_trace_folds_finds_a_ridge(depth_dir, tmp_path):
    out = tmp_path / "folds.json"
    r = _py("trace_folds.py", depth_dir / "depth.npy", depth_dir / "mask.npy", out)
    assert r.returncode == 0, r.stderr[-500:]
    assert "folds" in json.loads(out.read_text())


def test_the_starter_preset_has_every_setting_the_fit_searches():
    preset = json.loads((Path(__file__).resolve().parents[1] / "presets" / "starter.json").read_text())
    assert set(inverse_fit.plan_stages(preset)) == {"geometry", "light"}

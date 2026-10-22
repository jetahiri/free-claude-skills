"""Reference matcher: measurements, grade, failure cases and calibration. Tests run on the processor with generated
images; the ones that load a model run only with REFMATCH_TEST_MODELS=1."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refmatch  # noqa: E402
import refmatch_calibrate  # noqa: E402
import refmatch_store  # noqa: E402

needs_models = pytest.mark.skipif(os.environ.get("REFMATCH_TEST_MODELS") != "1",
                                  reason="set REFMATCH_TEST_MODELS=1 to run tests that load models")


def test_palette_shares_sum_to_one(REF):
    rgb = refmatch.load_rgb(REF, [])
    pal, labels = refmatch.palette(refmatch.rgb2lab(refmatch.resize(rgb, 96, 171)), 6)
    assert len(pal) == 6
    assert abs(sum(c["share"] for c in pal) - 1) < 0.01
    assert pal[0]["share"] >= pal[-1]["share"]
    assert labels.max() == 5


def test_identity_lut_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.random((40, 30, 3)).astype(np.float32)
    lut = refmatch.fit_lut(img, img)
    refmatch.write_cube(lut, tmp_path / "id.cube", "id")
    back = refmatch.read_cube(tmp_path / "id.cube")
    assert back.shape == (refmatch.LUT_SIZE ** 3, 3)
    assert np.abs(refmatch.apply_lut(img, back) - img).max() < 0.02


def test_grade_moves_render_toward_reference(tmp_path, REF, fix):
    ref = refmatch.load_rgb(REF, [])
    w, h = refmatch.work_size(ref)
    ren = refmatch.fit_to(refmatch.load_rgb(fix / "dim.png", []), w, h, [], "x")
    g = refmatch.grade(refmatch.resize(ref, w, h), ren, tmp_path, "t")
    assert g["de_after_lut"] < g["de_before"]
    assert (tmp_path / "grade.cube").read_text().startswith('TITLE "refmatch t"')


def test_tiny_image_stops_with_reason(tmp_path):
    Image.new("RGB", (10, 10), (0, 120, 100)).save(tmp_path / "tiny.png")
    with pytest.raises(refmatch.RefmatchError, match="too small"):
        refmatch.load_rgb(tmp_path / "tiny.png", [])


def test_missing_and_unreadable_images(tmp_path):
    with pytest.raises(refmatch.RefmatchError, match="not found"):
        refmatch.load_rgb(tmp_path / "nope.png", [])
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(refmatch.RefmatchError, match="cannot read"):
        refmatch.load_rgb(tmp_path / "bad.png", [])


def test_transparent_png_is_flattened_with_note(tmp_path, fix):
    im = Image.open(fix / "clay-rgba.png").convert("RGBA")
    a = np.asarray(im).copy()
    a[:40, :, 3] = 0
    Image.fromarray(a).save(tmp_path / "t.png")
    notes = []
    rgb = refmatch.load_rgb(tmp_path / "t.png", notes, bg=(0.1, 0.4, 0.35))
    assert rgb.shape[2] == 3 and "transparency" in notes[0]
    assert np.allclose(rgb[0, 0], (0.1, 0.4, 0.35), atol=0.01)


def test_aspect_mismatch_is_cropped_not_stretched():
    notes = []
    out = refmatch.fit_to(np.zeros((100, 300, 3), np.float32), 50, 100, notes, "wide.png")
    assert out.shape == (100, 50, 3) and "centre-cropped" in notes[0]


def test_auto_without_a_card_says_depth_ran_on_the_processor(tmp_path, REF, monkeypatch):
    monkeypatch.setattr(refmatch.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, 'RESULT {"device": "cpu", "images": []}\n', ""))
    notes = []
    res = refmatch.run_depth([REF], tmp_path, notes)
    assert res["device"] == "cpu" and any("no NVIDIA card" in n for n in notes)


def test_model_failure_skips_depth_with_note(tmp_path, REF, monkeypatch):
    monkeypatch.setattr(refmatch.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "OSError: cannot download model"))
    notes = []
    assert refmatch.run_depth([REF], tmp_path, notes) == {}
    assert notes[-1].startswith("depth skipped") and any("cannot download" in n for n in notes)


def test_cuda_asked_for_without_a_card_is_one_plain_line(tmp_path, REF):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this Python has an NVIDIA card; the no-card case cannot be forced here")
    notes = []
    assert refmatch.run_depth([REF], tmp_path, notes, device="cuda") == {}
    assert any("cannot see an NVIDIA card" in n for n in notes) and notes[-1].startswith("depth skipped")


@needs_models
def test_depth_model_runs_end_to_end(tmp_path, REF):
    notes = []
    res = refmatch.run_depth([REF], tmp_path, notes, device="cpu")
    assert res["device"] == "cpu" and (tmp_path / "reference.depth.npy").is_file()


@needs_models
def test_perceptual_score_with_the_real_model(REF, fix):
    notes = []
    d = refmatch.lpips_distance(refmatch.load_rgb(REF, []), refmatch.load_rgb(fix / "close.png", []), notes)
    assert d is not None and 0 <= d < 0.5, notes


def test_perceptual_model_missing_makes_the_result_incomplete(tmp_path, bars, REF, fix, monkeypatch):
    monkeypatch.setattr(refmatch, "lpips_distance", lambda a, b, notes: notes.append("perceptual score unavailable") or None)
    [r] = refmatch.run(REF, [fix / "close.png"], "3d-background", depth=False, out_root=tmp_path, record=False)
    assert r["passed"] is None and any("missing perceptual" in f for f in r["fails"])


def test_missing_required_score_is_incomplete_not_pass():
    bar = {"weights": {"structure": 1.0, "perceptual": 1.0}, "min": {"perceptual": 0.5}, "combined_min": 10}
    combined, passed, fails = refmatch.judge({"structure": 0.9, "perceptual": None}, bar)
    assert passed is None and combined == 90.0 and "missing perceptual" in fails[-1]


def test_proof_ranking_and_records(tmp_path, bars, REF, fix):
    renders = [fix / n for n in ("wrong.png", "close.png", "dim.png")]
    res = refmatch.run(REF, renders, "3d-background", build=1, label="t", depth=False, out_root=tmp_path)
    order = [Path(r["render"]).name for r in res]
    assert order[0] == "close.png" and order[-1] == "wrong.png"
    for r in res:
        out = Path(r["out"])
        for f in ("report.html", "side-by-side.png", "grade.cube", "graded.png", "report.json"):
            assert (out / f).is_file()
        assert r["match_id"] and r["norm"]["clarity"] is not None
    assert refmatch_store.db_path(bars).is_file()


def test_cube_lines_are_red_fastest(tmp_path):
    n = refmatch.LUT_SIZE
    lut = refmatch.fit_lut(np.zeros((4, 4, 3), np.float32), np.zeros((4, 4, 3), np.float32))  # shape only
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    ident = np.stack([r.ravel(), gg.ravel(), b.ravel()], 1)
    refmatch.write_cube(ident, tmp_path / "i.cube", "i")
    rows = [l.split() for l in (tmp_path / "i.cube").read_text().splitlines()[4:]]
    # .cube spec: the first index (red) changes fastest, then green, then blue
    assert [float(v) for v in rows[1]] == pytest.approx([1 / (n - 1), 0, 0], abs=1e-6)
    assert [float(v) for v in rows[n]] == pytest.approx([0, 1 / (n - 1), 0], abs=1e-6)
    assert [float(v) for v in rows[n * n]] == pytest.approx([0, 0, 1 / (n - 1)], abs=1e-6)
    # a LUT that swaps red and blue, applied to pure red, must give pure blue
    swap = ident[:, ::-1].copy()
    out = refmatch.apply_lut(np.array([[[1.0, 0.0, 0.0]]], np.float32), swap)
    assert out[0, 0] == pytest.approx([0, 0, 1], abs=1e-5)
    assert lut.shape == (n ** 3, 3)


def test_grade_skips_when_it_would_not_help(tmp_path, REF):
    ref = refmatch.load_rgb(REF, [])
    w, h = refmatch.work_size(ref)
    small = refmatch.resize(ref, w, h)
    g = refmatch.grade(small, small, tmp_path, "same")
    assert g["skipped"] and g["de_after_lut"] == g["de_before"]


def test_header_size_refused_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(refmatch, "MAX_PIXELS", 5000)
    Image.new("RGB", (100, 80)).save(tmp_path / "big.png")
    with pytest.raises(refmatch.RefmatchError, match="too large"):
        refmatch.load_rgb(tmp_path / "big.png", [])


def test_extreme_shape_refused(tmp_path):
    Image.new("RGB", (40, 2000)).save(tmp_path / "strip.png")
    with pytest.raises(refmatch.RefmatchError, match="too extreme"):
        refmatch.load_rgb(tmp_path / "strip.png", [])


def test_label_with_colon_makes_a_safe_folder(tmp_path, bars, REF, fix):
    [r] = refmatch.run(REF, [fix / "close.png"], "3d-background", label="v2: final/look?", depth=False,
                       out_root=tmp_path, record=False)
    assert Path(r["out"]).is_dir() and ":" not in Path(r["out"]).parent.name


def test_exif_rotation_is_applied(tmp_path):
    im = Image.new("RGB", (60, 40), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: display rotated 90 degrees
    im.save(tmp_path / "side.jpg", exif=exif)
    assert refmatch.load_rgb(tmp_path / "side.jpg", []).shape[:2] == (60, 40)


_kind = {"name": "calib-test"}


def _calib_kind(bars, combined_min=50):
    cfg = json.loads(bars.read_text())
    cfg["kinds"][_kind["name"]] = {"weights": {"structure": 1.0}, "min": {}, "combined_min": combined_min}
    bars.write_text(json.dumps(cfg))


def _match(structure, passed):
    with refmatch_store.session(refmatch.BARS_FILE) as con:
        return con.execute(
            "INSERT INTO reference_matches (created_at, kind, reference, combined, metrics, passed) VALUES (?,?,?,?,?,?)",
            ("2026-10-17T14:00:00", _kind["name"], "r", structure * 100, json.dumps({"norm": {"structure": structure}}),
             passed)).lastrowid


def _rows(sql, args):
    with refmatch_store.session(refmatch.BARS_FILE) as con:
        return con.execute(sql, args).fetchall()


def _bar(bars):
    return json.loads(bars.read_text())["kinds"][_kind["name"]]["combined_min"]


def test_reject_of_a_pass_raises_bar_and_logs_words(bars):
    _calib_kind(bars)
    m = _match(0.8, 1)
    out = refmatch_calibrate.verdict(m, "reject", "too grainy", "blind")
    assert out["agrees"] is False and _bar(bars) == 81.0 and not out["clashes"]
    log = _rows("SELECT reason FROM refmatch_calibration WHERE match_id=?", [m])
    assert "too grainy" in log[0]["reason"] and "blind" in log[0]["reason"]


def test_bar_stays_between_rejected_and_accepted(bars):
    _calib_kind(bars, combined_min=80)
    rej = _match(0.718, 0)
    refmatch_calibrate.verdict(rej, "reject", "no")
    acc = _match(0.72, 0)
    refmatch_calibrate.verdict(acc, "accept", "yes")
    assert 71.8 < _bar(bars) <= 72.0
    later = _match(0.75, 1)
    refmatch_calibrate.verdict(later, "reject", "no after all")
    # 0.75 rejected above an accepted 0.72: no bar agrees with both, so it is a logged conflict, never silent
    kinds = [r["change"] for r in _rows("SELECT change FROM refmatch_calibration WHERE kind=? ORDER BY id", [_kind["name"]])]
    assert "conflict" in kinds and "clash" in kinds


def test_second_verdict_keeps_the_first_in_notes(bars):
    _calib_kind(bars)
    m = _match(0.6, 1)
    refmatch_calibrate.verdict(m, "accept", "good")
    refmatch_calibrate.verdict(m, "reject", "on reflection, lumpy")
    row = _rows("SELECT verdict, notes FROM reference_matches WHERE id=?", [m])[0]
    assert row["verdict"] == "reject" and any('accept "good"' in n for n in json.loads(row["notes"]))
    assert _bar(bars) > 60


def test_agreement_report(bars):
    _calib_kind(bars)
    refmatch_calibrate.verdict(_match(0.9, 1), "accept", "yes")
    refmatch_calibrate.verdict(_match(0.3, 0), "reject", "no")
    agg = refmatch_calibrate.agreement(_kind["name"])[_kind["name"]]
    assert agg["scored"] == 2 and agg["agreement"] == 1.0


def test_unknown_match_number_is_a_plain_error(bars):
    with pytest.raises(refmatch.RefmatchError, match="no reference match"):
        refmatch_calibrate.verdict(999, "accept")


def test_records_file_is_created_on_first_use_beside_the_bars(tmp_path, bars, monkeypatch):
    monkeypatch.delenv("REFMATCH_DB")
    assert refmatch_store.db_path(bars) == bars.with_name("refmatch_records.sqlite")
    with refmatch_store.session(bars):
        pass
    assert bars.with_name("refmatch_records.sqlite").is_file()


def test_command_line_run_verdict_and_bars(tmp_path, bars, REF, fix, capsys):
    out = tmp_path / "out"
    code = refmatch.main(["run", "--ref", str(REF), "--render", str(fix / "close.png"), "--kind", "3d-clay",
                          "--no-depth", "--out", str(out), "--label", "cli"])
    printed = capsys.readouterr().out
    assert code == 0 and "match #1" in printed and "report:" in printed
    assert refmatch.main(["verdict", "1", "accept", "--words", "looks right", "--seen", "blind"]) == 0
    assert '"verdict": "accept"' in capsys.readouterr().out
    assert refmatch.main(["bars", "--kind", "3d-clay"]) == 0
    assert refmatch.main(["agreement"]) == 0


def test_command_line_stops_with_one_plain_line(tmp_path, bars, capsys):
    code = refmatch.main(["run", "--ref", str(tmp_path / "nope.png"), "--render", str(tmp_path / "x.png"),
                          "--kind", "3d-clay", "--out", str(tmp_path)])
    assert code == 2 and "image not found" in capsys.readouterr().err


def test_script_runs_standalone_from_any_folder(tmp_path):
    r = subprocess.run([sys.executable, str(Path(refmatch.__file__)), "bars", "--kind", "web-page"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "combined_min" in r.stdout

import json

import numpy as np

import inverse_fit as f

PARAMS = [("geometry.smooth", 2.0, 10.0), ("lights.1.power", 1200.0, 5000.0)]


def test_unit_round_trip_and_clipping():
    vals = {"geometry.smooth": 6.0, "lights.1.power": 2600.0}
    back = f.from_unit(f.to_unit(vals, PARAMS), PARAMS)
    assert back == {"geometry.smooth": 6.0, "lights.1.power": 2600.0}
    out = f.from_unit(np.array([-0.5, 1.7]), PARAMS)
    assert out == {"geometry.smooth": 2.0, "lights.1.power": 5000.0}
    assert f.to_unit({"geometry.smooth": 50.0, "lights.1.power": 0.0}, PARAMS).tolist() == [1.0, 0.0]


def test_put_get_reach_into_lists():
    S = {"lights": [{"power": 1}, {"power": 2}], "geometry": {"smooth": 6}}
    f.put(S, "lights.1.power", 9)
    assert f.get(S, "lights.1.power") == 9 and S["lights"][0]["power"] == 1


def test_set_args_halves_grid_cell_settings_only():
    args = f.set_args({"geometry.smooth": 6.0, "geometry.curl": 0.4}, True, [600, 1070])
    pairs = [args[i + 1] for i in range(0, len(args), 2)]
    assert pairs[0] == "geometry.grid=[300, 535]"
    assert "geometry.smooth=3.0" in pairs and "geometry.curl=0.4" in pairs
    full = f.set_args({"geometry.smooth": 6.0}, False, [600, 1070])
    assert full == ["--set", "geometry.smooth=6.0"]
    assert json.loads(pairs[0].split("=", 1)[1]) == [300, 535]


def test_es_stage_stops_when_no_gain():
    calls = []

    def flat(x):
        calls.append(x)
        return 50.0

    _, best, gens = f.es_stage(flat, np.full(3, 0.5), 50.0, pop=4, max_gens=10, rng=np.random.default_rng(0))
    assert best == 50.0 and gens == f.PATIENCE and len(calls) == 4 * f.PATIENCE


def test_es_stage_climbs_and_stays_in_box():
    target = np.array([0.9, 0.1])
    seen = []

    def bowl(x):
        seen.append(x)
        return 100 - 100 * float(((x - target) ** 2).sum())

    x, best, _ = f.es_stage(bowl, np.array([0.2, 0.8]), bowl(np.array([0.2, 0.8])), pop=6, max_gens=30,
                            rng=np.random.default_rng(1))
    assert best > 95 and np.abs(x - target).max() < 0.2
    assert all((s >= 0).all() and (s <= 1).all() for s in seen)


def test_es_stage_small_gains_count_as_stale():
    scores = iter([50.1, 50.2] + [0.0] * 100)

    def creep(x):
        return next(scores)

    _, best, gens = f.es_stage(creep, np.full(2, 0.5), 50.0, pop=1, max_gens=10, rng=np.random.default_rng(2))
    assert best == 50.2 and gens == f.PATIENCE


def test_finish_is_deterministic_and_in_range():
    rgb = np.random.default_rng(3).random((40, 30, 3)).astype(np.float32)
    a = f.apply_finish(rgb, None, 0.5, 0.02)
    b = f.apply_finish(rgb, None, 0.5, 0.02)
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() <= 1
    assert np.array_equal(f.apply_finish(rgb, None, 0.0, 0.0), rgb)


def test_sharpen_changes_the_picture_and_grain_stays_off_the_background():
    rgb = np.zeros((40, 30, 3), np.float32) + 0.3
    rgb[10:30, 8:22] = 0.7
    sharp = f.apply_finish(rgb, None, 1.0, 0.0)
    assert not np.array_equal(sharp, rgb)
    forms = np.zeros((40, 30), np.float32)
    forms[10:30, 8:22] = 1
    grained = f.apply_finish(rgb, None, 0.0, 0.05, forms)
    assert np.array_equal(grained[forms == 0], rgb[forms == 0])
    assert grained[forms == 1].std() > 0.01


def test_aborted_fit_still_writes_best_and_preset(tmp_path, monkeypatch):
    preset = {"image": {"depth": "d.npy"}, "geometry": {"grid": [8, 8], "smooth": 6.0, "curl": 0.3, "curl_width": 0.2,
              "thickness": 1.5, "gap_depth": 0.0}, "material": {"roughness": 0.5, "subsurface_scale": 1.0, "sheen": 0.0},
              "lights": [{"power": 900}, {"power": 3000}, {"power": 200}], "world": {"strength": 0.3},
              "camera": {"fstop": 1.0}}
    sp = tmp_path / "p.json"
    sp.write_text(json.dumps(preset))
    calls = {"n": 0}

    def fake_render(settings, over, out, a, grid):
        calls["n"] += 1
        if calls["n"] > 1:
            return None
        from PIL import Image
        Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(out)
        return np.full((8, 8, 3), 0.5, np.float32)

    class FakeScorer:
        def __init__(self, ref):
            pass

        def lut(self, rgb, mode="histogram"):
            return None

    monkeypatch.setattr(f, "render", fake_render)
    monkeypatch.setattr(f, "Scorer", FakeScorer)
    monkeypatch.setattr(f, "fit_finish", lambda sc, rgb: (60.0, {}, {"lut": "none", "sharpen": 0.0, "grain": 0.0}, None))
    monkeypatch.setattr(f, "forms_mask", lambda sc, rgb: np.ones(rgb.shape[:2], np.float32))
    monkeypatch.setattr(f, "load_render", lambda p: np.full((8, 8, 3), 0.5, np.float32))
    f.main(["fit", "--settings", str(sp), "--ref", "r.jpg", "--out", str(tmp_path / "run"), "--pop", "2"])
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert "aborted" in summary and summary["best"] == 60.0
    assert (tmp_path / "p.fit.json").is_file() and list((tmp_path / "run").glob("best1-*.png"))


def test_look_stage_preset_missing_a_light_fails_before_rendering():
    preset = {"geometry": {"grid": [8, 8]}, "mesh": {"depth": 5.6, "scale": 1.0, "offset": [0, 0], "yaw": 0},
              "material": {"roughness": 0.7, "subsurface_scale": 1, "sheen": 0, "specular": 0.1, "grain_strength": 0.2},
              "lights": [{"power": 1}, {"power": 1}, {"power": 1}], "world": {"strength": 0.5},
              "camera": {"fstop": 0.5}, "atmosphere": {"density": 0.02}, "glare": {"strength": 1, "threshold": 0.8}}
    try:
        f.plan_stages(preset)
    except SystemExit as e:
        assert "lights.3.power" in str(e)
    else:
        raise AssertionError("a look-stage preset with three lights must be refused up front")
    preset["lights"].append({"power": 1})
    assert "look" in f.plan_stages(preset)

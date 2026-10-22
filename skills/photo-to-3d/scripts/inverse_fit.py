"""Inverse rendering for the image-to-3D pipeline: adjust a preset's shape, material, light and
finishing settings until the Blender render matches the reference, scored by refmatch.

  python inverse_fit.py fit --settings <preset.json> --ref <reference image> --out <folder>
                            [--depths a.npy,b.npy] [--scale 0.25] [--samples 32] [--max-gens 4] [--pop 4]
  python inverse_fit.py finish --settings <fitted preset.json> --ref <reference> --render <png> --out <png>
  python inverse_fit.py score --ref <reference> --render <png>

Run fit as ONE unattended job; it calls Blender itself (set BLENDER to blender.exe when it is not on the PATH), writes log.jsonl (every candidate), summary.json, the best three finished renders and the
fitted preset <preset>.fit.json beside the original. The original preset is never changed.

Why the real renderer and not a differentiable one (Mitsuba 3, PyTorch): the fitted values must give the same picture
in Cycles (subsurface, AgX view, depth of field). Fitting through Blender itself transfers exactly; the gradient-based
part of the pipeline is the geometry fit in normals_from_ref.py. Search: a (1+lambda) evolution strategy in the unit
box of each stage's settings, step size grown on success and shrunk on failure; a stage stops after PATIENCE
generations without a MIN_GAIN point gain. Finishing (colour table to the reference, sharpening, grain) is fitted per
render in numpy, since it needs no re-render.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
# the scoring comes from the reference-matcher skill, installed as a sibling folder (skills/reference-matcher)
sys.path.insert(0, os.environ.get("REFMATCH_DIR", str(HERE.parent.parent / "reference-matcher" / "scripts")))
try:
    import refmatch as rm  # noqa: E402
except ImportError:
    raise SystemExit("inverse_fit needs the reference-matcher skill: install it next to this skill, or set REFMATCH_DIR "
                     "to its scripts folder")

KIND = "3d-background"
MIN_GAIN = 0.3
PATIENCE = 2
SHARPEN = (0.0, 0.5, 1.0)
GRAIN = (0.0, 0.006, 0.012, 0.02, 0.03, 0.045)
LUT_MODES = ("none", "histogram", "aligned")
EDGE_BAND = 6  # pixels at the 384-wide scoring size

# (setting path, low, high); the depth map is a separate categorical first stage
STAGES = {
    "geometry": [("geometry.smooth", 2.0, 10.0), ("geometry.curl", 0.0, 1.6), ("geometry.curl_width", 0.1, 0.9),
                 ("geometry.thickness", 0.8, 3.0), ("geometry.gap_depth", 0.0, 0.4)],
    "light": [("material.roughness", 0.2, 0.9), ("material.subsurface_scale", 0.2, 2.5), ("material.sheen", 0.0, 0.6),
              ("lights.0.power", 60.0, 2600.0), ("lights.1.power", 1200.0, 10000.0), ("lights.2.power", 0.0, 1500.0),
              ("world.strength", 0.0, 1.6), ("camera.fstop", 0.08, 2.5)],
}
# a preset with a generated mesh: placement replaces the depth-surface shape stage
MESH_STAGE = [("mesh.depth", 4.6, 7.0), ("mesh.scale", 0.75, 1.3), ("mesh.offset.0", -0.2, 0.2),
              ("mesh.offset.1", -0.2, 0.2), ("mesh.yaw", -30.0, 30.0)]
# a preset with atmosphere and glow: fill light, haze, glare, grain
LOOK_STAGE = [("lights.3.power", 0.0, 600.0), ("atmosphere.density", 0.0, 0.04), ("glare.strength", 0.0, 1.5),
              ("glare.threshold", 0.3, 1.0), ("material.specular", 0.0, 0.4), ("material.grain_strength", 0.0, 0.6)]
# settings measured in grid cells: halve them when the fit renders on a half-size grid
GRID_CELLS = ("geometry.smooth", "geometry.gap_soften")


class FitAborted(Exception):
    pass


def blender_exe() -> str:
    found = os.environ.get("BLENDER") or shutil.which("blender")
    if found:
        return found
    hits = sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files"), "Blender Foundation").glob("Blender */blender.exe"))
    hits += [p for p in (Path("/Applications/Blender.app/Contents/MacOS/Blender"),) if p.is_file()]
    if not hits:
        raise SystemExit("Blender not found: install it from blender.org (the tested version is in the README), then set BLENDER "
                         "to the full path of blender.exe, or add it to the PATH")
    return str(hits[-1])


def get(S: dict, key: str):
    for p in key.split("."):
        S = S[int(p)] if isinstance(S, list) else S[p]
    return S


def put(S: dict, key: str, val) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        S = S[int(p)] if isinstance(S, list) else S[p]
    S[int(parts[-1]) if isinstance(S, list) else parts[-1]] = val


def to_unit(vals: dict, params: list) -> np.ndarray:
    return np.array([(vals[k] - lo) / (hi - lo) for k, lo, hi in params]).clip(0, 1)


def from_unit(x: np.ndarray, params: list) -> dict:
    x = np.clip(x, 0, 1)
    return {k: round(float(lo + xi * (hi - lo)), 4) for xi, (k, lo, hi) in zip(x, params)}


def set_args(over: dict, half_grid: bool, grid: list) -> list:
    """--set overrides for image_to_3d.py; the fit grid is half size, so cell-measured settings are halved too."""
    out = []
    if half_grid:
        out += ["--set", f"geometry.grid={json.dumps([grid[0] // 2, grid[1] // 2])}"]
    for k, v in over.items():
        if half_grid and k in GRID_CELLS:
            v = v / 2
        out += ["--set", f"{k}={json.dumps(v)}"]
    return out


class Scorer:
    def __init__(self, ref_path: Path):
        self.cfg = rm.load_bars()["kinds"][KIND]
        self.ref_full = rm.load_rgb(ref_path, [])
        self.w, self.h = rm.work_size(self.ref_full)
        self.ref = rm.resize(self.ref_full, self.w, self.h)

    def score(self, ren_full: np.ndarray) -> tuple[float, dict]:
        """Scores the exact 8-bit picture that would be saved, so a fit score equals a later refmatch run."""
        ren_full = to_8bit(ren_full).astype(np.float32) / 255
        notes: list = []
        ren = rm.fit_to(ren_full, self.w, self.h, notes, "render")
        _, norm, _ = rm.measure(self.ref, ren, notes)
        norm["clarity"] = rm.clarity(self.ref_full, ren_full)[1]
        combined, _, _ = rm.judge(norm, self.cfg)
        return combined or 0.0, norm

    def lut(self, ren_full: np.ndarray, mode: str = "histogram") -> np.ndarray | None:
        """A colour table onto the reference; None when it would not lower the colour difference.
        histogram: refmatch's own grade (match the colour spread in Lab). aligned: the render is built through the
        photo's framing, so each blurred render pixel is fitted straight to the same blurred photo pixel."""
        ren = rm.fit_to(ren_full, self.w, self.h, [], "render")
        lab_r, lab_s = rm.rgb2lab(self.ref), rm.rgb2lab(ren)
        if mode == "aligned":
            b = lambda x: rm.cv2.GaussianBlur(x, (0, 0), 3)
            # leave out a band around both outlines: there a blurred render pixel can sit over the photo's warm rim
            # or background, and fitting those pairs maps the render's edge colours to saturated orange
            ms = rm.silhouette(lab_s, hint=rm.background_lab(lab_r))
            mr = rm.silhouette(lab_r)
            edge = lambda m: rm.cv2.morphologyEx(m.astype(np.uint8), rm.cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            band = rm.cv2.dilate(edge(ms) | edge(mr), np.ones((2 * EDGE_BAND + 1,) * 2, np.uint8)).astype(bool)
            keep = ~band & (ms == mr)
            table = rm.fit_lut(b(ren)[keep], b(self.ref)[keep])
        else:
            matched = rm.lab2rgb(rm.match_histograms(lab_s, lab_r, channel_axis=-1)).astype(np.float32)
            table = rm.fit_lut(ren, matched)
        before = float(rm.deltaE_ciede2000(lab_r, lab_s).mean())
        after = float(rm.deltaE_ciede2000(lab_r, rm.rgb2lab(rm.apply_lut(ren, table))).mean())
        return table if after < before else None


def load_render(path) -> np.ndarray:
    """RGB float 0..1 at the file's own precision: Blender writes 16-bit PNGs, and cutting them to 8 bits before a
    colour table stretches them shows as stepped bands in the smooth gradients."""
    a = rm.cv2.imread(str(path), rm.cv2.IMREAD_UNCHANGED)
    if a is None:
        return rm.load_rgb(path, [])
    a = a[..., :3][..., ::-1]
    return (a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)).copy()


def to_8bit(rgb: np.ndarray) -> np.ndarray:
    # no dither: against a very smooth photo even half a step of random dither scores as noise; banding only came
    # from stretching an already 8-bit render, which load_render avoids
    return np.clip(rgb * 255, 0, 255).round().astype(np.uint8)


def save_png(path, rgb: np.ndarray) -> None:
    Image.fromarray(to_8bit(rgb)).save(path)


def apply_finish(rgb: np.ndarray, table: np.ndarray | None, sharpen: float, grain: float,
                 forms: np.ndarray | None = None) -> np.ndarray:
    out = rm.apply_lut(rgb, table) if table is not None else rgb.astype(np.float32)
    if sharpen > 0:
        blur = np.asarray(Image.fromarray((out * 255).round().astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                          dtype=np.float32) / 255
        out = out + sharpen * (out - blur)
    if grain > 0:
        # fixed seed: the same settings always give the same picture, so scores are comparable
        n = np.random.default_rng(7).normal(0, grain, out.shape[:2]).astype(np.float32)
        # the photo's grain is a frosted texture on the forms; its background is smooth
        out = out + (n * forms if forms is not None else n)[..., None]
    return out.clip(0, 1)


def forms_mask(sc: Scorer, rgb: np.ndarray) -> np.ndarray:
    """Soft 0..1 mask of the forms at the render's own size (refmatch's silhouette, feathered)."""
    ren = rm.fit_to(rgb, sc.w, sc.h, [], "render")
    m = rm.silhouette(rm.rgb2lab(ren), hint=rm.background_lab(rm.rgb2lab(sc.ref))).astype(np.float32)
    m = rm.cv2.resize(rm.cv2.GaussianBlur(m, (0, 0), 1.5), (rgb.shape[1], rgb.shape[0]), interpolation=rm.cv2.INTER_LINEAR)
    return m.clip(0, 1)


def fit_finish(sc: Scorer, rgb: np.ndarray) -> tuple[float, dict, dict, np.ndarray | None]:
    """Colour table (none, histogram or aligned), then sharpening, then grain on the forms, each picked by score."""
    forms = forms_mask(sc, rgb)
    best = (-1.0, {}, {"lut": "none", "sharpen": 0.0, "grain": 0.0}, None)
    for mode in LUT_MODES:
        table = None if mode == "none" else sc.lut(rgb, mode)
        if mode != "none" and table is None:
            continue
        c, norm = sc.score(apply_finish(rgb, table, 0.0, 0.0))
        if c > best[0]:
            best = (c, norm, {"lut": mode, "sharpen": 0.0, "grain": 0.0}, table)
    table = best[3]
    for s in SHARPEN[1:]:
        f = dict(best[2], sharpen=s)
        c, norm = sc.score(apply_finish(rgb, table, s, 0.0))
        if c > best[0]:
            best = (c, norm, f, table)
    for g in GRAIN[1:]:
        f = dict(best[2], grain=g)
        c, norm = sc.score(apply_finish(rgb, table, f["sharpen"], g, forms))
        if c > best[0]:
            best = (c, norm, f, table)
    return best


def finish_table(sc: Scorer, rgb: np.ndarray, f: dict) -> np.ndarray | None:
    mode = {True: "histogram", False: "none"}.get(f["lut"], f["lut"])
    return None if mode == "none" else sc.lut(rgb, mode)


def render(settings: Path, over: dict, out: Path, a, grid: list, timeout: int = 900) -> np.ndarray | None:
    cmd = [blender_exe(), "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(HERE / "image_to_3d.py"),
           "--", "--settings", str(settings), "--stage", "final", "--out", str(out), "--scale", str(a.scale),
           "--samples", str(a.samples)] + set_args(over, True, grid)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not out.is_file():
        (out.parent / f"{out.stem}.err.txt").write_text(r.stdout[-3000:] + r.stderr[-3000:], encoding="utf-8")
        return None
    return load_render(out)


def es_stage(evaluate, x0: np.ndarray, best: float, pop: int, max_gens: int, rng) -> tuple[np.ndarray, float, int]:
    """(1+lambda) evolution strategy in the unit box; returns the best point, its score and generations run."""
    sigma, stale, x = 0.15, 0, x0.copy()
    for gen in range(1, max_gens + 1):
        gen_best = best
        for _ in range(pop):
            cand = np.clip(x + rng.normal(0, sigma, x.shape), 0, 1)
            c = evaluate(cand)
            if c > gen_best:
                gen_best, xb = c, cand
        if gen_best > best:
            gained = gen_best - best
            x, best, sigma = xb, gen_best, min(0.35, sigma * 1.3)
            stale = 0 if gained >= MIN_GAIN else stale + 1
        else:
            sigma, stale = max(0.03, sigma * 0.6), stale + 1
        if stale >= PATIENCE:
            return x, best, gen
    return x, best, max_gens


def plan_stages(S: dict) -> dict:
    stages = dict(STAGES)
    if S.get("mesh"):
        stages["geometry"] = MESH_STAGE
    if S.get("atmosphere") and S.get("glare"):
        stages["look"] = LOOK_STAGE
    # every searched setting must exist in the preset, or the fit dies hours in at that stage 
    missing = []
    for params in stages.values():
        for k, _, _ in params:
            try:
                get(S, k)
            except (KeyError, IndexError, TypeError, ValueError):
                missing.append(k)
    if missing:
        raise SystemExit(f"preset lacks settings the fit searches: {', '.join(missing)}")
    return stages


def cmd_fit(a) -> None:
    sp = Path(a.settings).resolve()
    S = json.loads(sp.read_text(encoding="utf-8"))
    stages = plan_stages(S)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sc = Scorer(Path(a.ref))
    log = (out / "log.jsonl").open("a", encoding="utf-8")
    rng = np.random.default_rng(7)
    state = {"n": 0, "fails": 0, "over": {}, "best": -1.0, "top": []}
    grid = S["geometry"]["grid"]

    def evaluate(over: dict, stage: str) -> float:
        state["n"] += 1
        png = out / f"c{state['n']:03d}.png"
        rgb = render(sp, over, png, a, grid)
        if rgb is None:
            state["fails"] += 1
            log.write(json.dumps({"n": state["n"], "stage": stage, "over": over, "failed": True}) + "\n")
            log.flush()
            if state["fails"] >= 3:
                raise FitAborted("three renders failed in a row; see the .err.txt files")
            return -1.0
        state["fails"] = 0
        c, norm, finish, _ = fit_finish(sc, rgb)
        log.write(json.dumps({"n": state["n"], "stage": stage, "over": over, "score": c, "norm": norm,
                              "finish": finish}) + "\n")
        log.flush()
        state["top"] = sorted(state["top"] + [(c, state["n"], over, finish)], key=lambda t: -t[0])[:3]
        if c > state["best"]:
            state["best"], state["over"] = c, dict(over)
        return c

    fixed0 = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.base)}
    state["over"] = dict(fixed0)
    summary = {"preset": str(sp), "base": fixed0, "stages": []}
    try:
        summary["start"] = evaluate(dict(fixed0), "start")
        # stage 1: which depth map (the original, or ones refined with the photo's normals)
        for d in [x for x in (a.depths or "").split(",") if x]:
            evaluate(fixed0 | {"image.depth": d}, "depth")
        summary["stages"].append({"stage": "depth", "best": state["best"], "over": dict(state["over"])})
        for name, params in stages.items():
            fixed = dict(state["over"])
            start_vals = {k: fixed.get(k, get(S, k)) for k, _, _ in params}
            x0 = to_unit(start_vals, params)
            _, best, gens = es_stage(lambda x: evaluate(fixed | from_unit(x, params), name), x0, state["best"],
                                     a.pop, a.max_gens, rng)
            summary["stages"].append({"stage": name, "best": best, "generations": gens, "over": dict(state["over"])})
    except FitAborted as e:
        # keep what was found so far: the best renders and the fitted preset are still written below
        summary["aborted"] = str(e)
        if not state["top"]:
            log.close()
            raise SystemExit(f"fit aborted before any render scored: {e}")
    log.close()
    base = summary.get("start", -1.0)

    for i, (c, n, _, finish) in enumerate(state["top"]):
        rgb = load_render(out / f"c{n:03d}.png")
        table = finish_table(sc, rgb, finish)
        save_png(out / f"best{i + 1}-c{n:03d}-{c:.1f}.png",
                 apply_finish(rgb, table, finish["sharpen"], finish["grain"], forms_mask(sc, rgb)))
    fitted = json.loads(sp.read_text(encoding="utf-8"))
    for k, v in state["over"].items():
        put(fitted, k, v)
    fitted["finish"] = {k: v for k, v in state["top"][0][3].items()}
    fitted["fit"] = {"from": sp.name, "reference": str(Path(a.ref).resolve()), "fit_score_small": state["best"],
                     "start_score_small": base, "candidates": state["n"], "scale": a.scale, "samples": a.samples}
    fp = sp.with_name((sp.stem if sp.stem.endswith(".fit") else sp.stem + ".fit") + ".json")
    fp.write_text(json.dumps(fitted, indent=1) + "\n", encoding="utf-8")
    summary |= {"best": state["best"], "over": state["over"], "candidates": state["n"], "fitted_preset": str(fp),
                "top": [{"score": c, "n": n} for c, n, _, _ in state["top"]]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("RESULT " + json.dumps({"start": base, "best": state["best"], "candidates": state["n"], "preset": str(fp)}))


def cmd_finish(a) -> None:
    """Apply a fitted preset's finish block to a (full-size) render: the colour table is re-fitted at that size."""
    S = json.loads(Path(a.settings).read_text(encoding="utf-8"))
    f = S.get("finish", {"lut": "none", "sharpen": 0.0, "grain": 0.0})
    sc = Scorer(Path(a.ref))
    rgb = load_render(a.render)
    if a.refit:
        c, norm, f, table = fit_finish(sc, rgb)
    else:
        table = finish_table(sc, rgb, f)
    img = apply_finish(rgb, table, f["sharpen"], f["grain"], forms_mask(sc, rgb))
    save_png(a.out, img)
    c, norm = sc.score(img)
    print("RESULT " + json.dumps({"score": c, "norm": norm, "finish": f}))


def cmd_score(a) -> None:
    c, norm = Scorer(Path(a.ref)).score(rm.load_rgb(a.render, []))
    print("RESULT " + json.dumps({"score": c, "norm": norm}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--settings", required=True)
    f.add_argument("--ref", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--depths", default="")
    f.add_argument("--base", action="append", default=[], help="key=json fixed for every candidate (e.g. tubes=null)")
    f.add_argument("--scale", type=float, default=0.25)
    f.add_argument("--samples", type=int, default=32)
    f.add_argument("--max-gens", type=int, default=4)
    f.add_argument("--pop", type=int, default=4)
    g = sub.add_parser("finish")
    g.add_argument("--settings", required=True)
    g.add_argument("--ref", required=True)
    g.add_argument("--render", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--refit", action="store_true", help="re-fit sharpening and grain at this render's size")
    s = sub.add_parser("score")
    s.add_argument("--ref", required=True)
    s.add_argument("--render", required=True)
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "finish": cmd_finish, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()

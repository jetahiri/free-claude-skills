"""Reference matcher: measure an inspiration image and score a recreation against it, so a faithful copy is proven
in numbers, never eyeballed.

  python refmatch.py run --ref R --render S [--render S2 ...] --kind 3d-background|3d-clay|web-page|video-frame
                         [--label L] [--no-depth] [--device auto|cuda|cpu] [--out DIR]
                                                                              everything, one report folder per render
  python refmatch.py palette --ref R [--k 6]                                  the reference's main colours and shares
  python refmatch.py grade --ref R --render S --out DIR                       graded render + grade.cube (33-point LUT)
  python refmatch.py verdict <match id> accept|reject [--words "..."] [--seen blind|seen|seen-possibly]
                                                                              your eye; tunes the bar when it disagrees
  python refmatch.py agreement [--kind K]                                     score-versus-verdict agreement over time
  python refmatch.py bars [--kind K]                                          the pass bars in force

What is code and what is a model: every measurement, the grade, the thresholds and the calibration are plain code
(k-means in Lab, CIEDE2000, SSIM, Otsu silhouette, Canny edges, polynomial colour fit). Two local models only: depth
(Depth Anything V2 Small, on an NVIDIA card when this Python can use one, else on the processor, see refmatch_depth.py)
and the perceptual score (LPIPS with AlexNet, on the processor; small enough at the working size).

Scores, each scaled 0..1 with higher meaning closer: silhouette (overlap of the forms-versus-background masks),
edges (F1 of edge maps within 2 px), structure (SSIM of lightness), colour (1 - mean CIEDE2000 / 25), perceptual
(1 - LPIPS), depth (Spearman correlation of the depth maps, mapped to 0..1). The combined score is their weighted mean
x 100 over the scores present. A render passes when every per-score minimum and the combined bar hold; a missing
required score (any with a minimum) makes the result incomplete, never a pass. Bars and weights per job kind live in
refmatch_bars.json and move only through verdicts (your eye wins), each move logged in refmatch_calibration.
"""
import argparse
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import re

from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_fill_holes
from scipy.stats import spearmanr
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity
from sklearn.cluster import KMeans

import refmatch_store

HERE = Path(__file__).resolve().parent
BARS_FILE = Path(os.environ.get("REFMATCH_BARS") or HERE / "refmatch_bars.json")
WORK_W = 384           # scoring width: enough for shape and colour, small enough for low main memory
MIN_SIDE = 32
MAX_PIXELS = 80_000_000
MAX_ASPECT = 8         # a 40x20000 strip would ask for ~74 million pixels at the scoring width
DE_SCALE = 25.0        # a mean CIEDE2000 of 25 is a different picture; 5 (colour 0.8) is close, under 2 is a match
LUT_SIZE = 33
METRICS = ("silhouette", "edges", "structure", "colour", "perceptual", "depth", "clarity")
CLARITY_MAX_W = 1024  # clarity is measured near full size: grain and blur vanish at the scoring width
_lpips_model = None


class RefmatchError(Exception):
    """A run cannot go on; the message is one plain line saying why."""


# ---------- images ----------

def load_rgb(path, notes: list, bg=None) -> np.ndarray:
    """RGB float 0..1. Transparent pixels are flattened over bg (the reference's background colour) with a note."""
    p = Path(path)
    if not p.is_file():
        raise RefmatchError(f"image not found: {p}")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    too_big = RefmatchError(f"image too large to measure safely (over {MAX_PIXELS // 1_000_000} million pixels): {p.name}")
    try:
        im = Image.open(p)
        # the size is in the header: refuse before decoding, since PIL only warns between 1x and 2x its limit
        if im.size[0] * im.size[1] > MAX_PIXELS:
            raise too_big
        im.load()
        im = ImageOps.exif_transpose(im)
    except RefmatchError:
        raise
    except Image.DecompressionBombError:
        raise too_big
    except Exception as e:
        raise RefmatchError(f"cannot read image {p.name}: {e}")
    if max(im.size) > MAX_ASPECT * min(im.size):
        raise RefmatchError(f"image shape too extreme to compare ({im.size[0]}x{im.size[1]}, over {MAX_ASPECT}:1): {p.name}")
    if min(im.size) < MIN_SIDE:
        raise RefmatchError(f"image too small to measure ({im.size[0]}x{im.size[1]}, need {MIN_SIDE} px a side): {p.name}")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255
        a = rgba[..., 3:4]
        if (a < 1).any():
            under = np.asarray(bg if bg is not None else (0.5, 0.5, 0.5), dtype=np.float32)
            notes.append(f"{p.name} has transparency ({(a < 1).mean():.0%} of pixels); flattened over "
                         f"{'the reference background' if bg is not None else 'mid grey'}")
            return rgba[..., :3] * a + under * (1 - a)
        return rgba[..., :3]
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255


def resize(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((w, h), Image.LANCZOS),
                      dtype=np.float32) / 255


def fit_to(rgb: np.ndarray, w: int, h: int, notes: list, name: str) -> np.ndarray:
    """Centre-crop to the target aspect, then resize: a stretched render would score shape it does not have."""
    ih, iw = rgb.shape[:2]
    want, have = w / h, iw / ih
    if abs(have / want - 1) > 0.02:
        notes.append(f"{name} aspect {iw}x{ih} differs from the reference; centre-cropped to match")
        if have > want:
            cw = round(ih * want)
            x0 = (iw - cw) // 2
            rgb = rgb[:, x0:x0 + cw]
        else:
            ch = round(iw / want)
            y0 = (ih - ch) // 2
            rgb = rgb[y0:y0 + ch]
    return resize(rgb, w, h)


def work_size(rgb: np.ndarray) -> tuple[int, int]:
    h, w = rgb.shape[:2]
    return WORK_W, max(MIN_SIDE, round(WORK_W * h / w))


# ---------- measurements ----------

def palette(lab: np.ndarray, k: int = 6) -> tuple[list[dict], np.ndarray]:
    """Main colours of an image in Lab, largest share first, plus each pixel's colour index (for per-region scores)."""
    X = lab.reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample = X[rng.choice(len(X), min(len(X), 20000), replace=False)]
    k = max(1, min(k, len(np.unique(sample.round(1), axis=0))))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(sample)
    labels = km.predict(X)
    share = np.bincount(labels, minlength=k) / len(labels)
    order = np.argsort(-share)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    pal = []
    for c in order:
        rgb = (lab2rgb(km.cluster_centers_[c][None, None])[0, 0] * 255).round().clip(0, 255).astype(int)
        pal.append({"hex": "#%02x%02x%02x" % tuple(rgb), "lab": [round(float(v), 1) for v in km.cluster_centers_[c]],
                    "share": round(float(share[c]), 4)})
    return pal, remap[labels].reshape(lab.shape[:2])


def background_lab(lab: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
    """The background colour from the outer 3% band, split into two colour groups (forms may touch an edge). With no
    hint the larger group wins (the reference); with a hint (the reference's background) the group nearest to it wins,
    because a render whose forms fill most of the border would otherwise have its forms taken for background."""
    h, w = lab.shape[:2]
    b = max(2, round(min(h, w) * 0.03))
    band = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                           lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    if len(np.unique(band.round(0), axis=0)) < 2:
        return np.median(band, 0)
    km = KMeans(n_clusters=2, n_init=2, random_state=0).fit(band)
    if hint is not None:
        big = int(np.argmin(np.linalg.norm(km.cluster_centers_ - np.asarray(hint), axis=1)))
    else:
        big = np.argmax(np.bincount(km.labels_, minlength=2))
    return np.median(band[km.labels_ == big], 0)


def silhouette(lab: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
    """Forms versus background: colour distance from the background colour, split at the Otsu threshold (floor 6), then
    closed and hole-filled, so shadowed hollows enclosed by a form (close to the background colour) count as form."""
    dist = deltaE_ciede2000(lab, np.broadcast_to(background_lab(lab, hint), lab.shape))
    t = threshold_otsu(dist) if dist.max() - dist.min() > 1e-6 else 0
    m = (dist > max(6.0, float(t))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    return binary_fill_holes(m)


def edges(lab: np.ndarray) -> np.ndarray:
    L = cv2.GaussianBlur((lab[..., 0] * 2.55).clip(0, 255).astype(np.uint8), (5, 5), 1.2)
    med = float(np.median(L))
    return cv2.Canny(L, max(5, int(0.66 * med * 0.5)), max(20, int(1.33 * med * 0.5))) > 0


def edge_f1(er: np.ndarray, es: np.ndarray, tol: int = 2) -> float | None:
    if er.sum() == 0 and es.sum() == 0:
        return None
    kern = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    dr = cv2.dilate(er.astype(np.uint8), kern) > 0
    ds = cv2.dilate(es.astype(np.uint8), kern) > 0
    precision = (es & dr).sum() / max(1, es.sum())
    recall = (er & ds).sum() / max(1, er.sum())
    return float(0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))


def _ratio(a: float, b: float) -> float:
    return 1.0 if max(a, b) < 1e-9 else float(min(a, b) / max(a, b))


def clarity_stats(rgb: np.ndarray) -> dict:
    """Noise in flat areas, edge sharpness along the main edges and fine-grain energy, on lightness."""
    L = rgb2lab(rgb)[..., 0].astype(np.float32)
    resid = L - cv2.GaussianBlur(L, (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1))
    smooth_grad = cv2.GaussianBlur(grad, (0, 0), 3)
    flat = smooth_grad < np.percentile(smooth_grad, 40)
    main = edges(rgb2lab(rgb))
    lap = np.abs(cv2.Laplacian(L, cv2.CV_32F))
    f = np.abs(np.fft.rfft2(L - L.mean())) ** 2
    fy = np.fft.fftfreq(L.shape[0])[:, None]
    fx = np.fft.rfftfreq(L.shape[1])[None, :]
    r = np.hypot(fy, fx)
    total = f[(r > 0.01)].sum() or 1.0
    return {"noise": float(resid[flat].std()) if flat.any() else 0.0,
            "sharpness": float(lap[main].mean()) if main.any() else float(lap.mean()),
            "grain": float(f[(r > 0.25) & (r <= 0.5)].sum() / total)}


def clarity(ref_full: np.ndarray, ren_full: np.ndarray) -> tuple[dict, float]:
    """Each clarity measure as a closeness ratio (1 = same as the reference), at a common size near full resolution."""
    h, w = ref_full.shape[:2]
    cw = min(CLARITY_MAX_W, w, ren_full.shape[1])
    ch = round(cw * h / w)
    a, b = clarity_stats(resize(ref_full, cw, ch)), clarity_stats(fit_to(ren_full, cw, ch, [], ""))
    parts = {k: round(_ratio(a[k], b[k]), 3) for k in a}
    raw = {f"{k}_ref": round(a[k], 4) for k in a} | {f"{k}_render": round(b[k], 4) for k in b} | parts
    return raw, round(float(np.mean(list(parts.values()))), 3)


def lpips_distance(a: np.ndarray, b: np.ndarray, notes: list) -> float | None:
    global _lpips_model
    try:
        import torch
        import lpips
        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
        t = lambda x: torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).float() * 2 - 1
        with torch.no_grad():
            return float(_lpips_model(t(a), t(b)).item())
    except Exception as e:
        notes.append(f"perceptual score unavailable ({type(e).__name__}: {str(e)[:120]})")
        return None


# ---------- colour grade ----------

def _poly(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    one = np.ones_like(r)
    return np.stack([one, r, g, b, r * r, g * g, b * b, r * g, r * b, g * b,
                     r ** 3, g ** 3, b ** 3, r * r * g, r * r * b, g * g * r, g * g * b, b * b * r, b * b * g, r * g * b], 1)


def fit_lut(src: np.ndarray, dst: np.ndarray, size: int = LUT_SIZE) -> np.ndarray:
    """A 3D LUT (size^3 x 3, red fastest) mapping src colours to dst: a third-order polynomial colour fit (ridge),
    sampled on the lattice. Smooth by construction, so colours the render does not contain still get a sane value."""
    X, Y = _poly(src.reshape(-1, 3).astype(np.float64)), dst.reshape(-1, 3).astype(np.float64)
    A = X.T @ X + 1e-3 * np.eye(X.shape[1])
    coef = np.linalg.solve(A, X.T @ Y)
    g = np.linspace(0, 1, size)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    lattice = np.stack([r.ravel(), gg.ravel(), b.ravel()], 1)
    return (_poly(lattice) @ coef).clip(0, 1)


def write_cube(lut: np.ndarray, path: Path, title: str) -> None:
    size = round(len(lut) ** (1 / 3))
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in lut]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def read_cube(path) -> np.ndarray:
    rows = [l.split() for l in Path(path).read_text(encoding="ascii").splitlines()]
    return np.array([[float(v) for v in r] for r in rows if len(r) == 3 and r[0][0] in "0123456789.-"])


def apply_lut(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear lookup of a red-fastest LUT, the order .cube files use."""
    n = round(len(lut) ** (1 / 3))
    cube = lut.reshape(n, n, n, 3)  # [b, g, r]
    p = rgb.clip(0, 1) * (n - 1)
    i0 = np.floor(p).astype(int).clip(0, n - 2)
    f = p - i0
    out = np.zeros_like(rgb, dtype=np.float64)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = ((f[..., 0] if dr else 1 - f[..., 0]) * (f[..., 1] if dg else 1 - f[..., 1])
                     * (f[..., 2] if db else 1 - f[..., 2]))
                out += w[..., None] * cube[i0[..., 2] + db, i0[..., 1] + dg, i0[..., 0] + dr]
    return out.astype(np.float32)


def grade(ref: np.ndarray, render: np.ndarray, out: Path, label: str) -> dict:
    """Colour transfer render -> reference: histogram matching in Lab (per channel), then the LUT that reproduces it."""
    lab_r, lab_s = rgb2lab(ref), rgb2lab(render)
    matched = lab2rgb(match_histograms(lab_s, lab_r, channel_axis=-1)).astype(np.float32)
    lut = fit_lut(render, matched)
    graded = apply_lut(render, lut)
    before = float(deltaE_ciede2000(lab_r, lab_s).mean())
    after = float(deltaE_ciede2000(lab_r, rgb2lab(graded)).mean())
    skipped = after >= before
    if skipped:
        # a render already close in colour can be nudged further by a global grade: hand back an identity LUT
        g = np.linspace(0, 1, LUT_SIZE)
        b, gg, r = np.meshgrid(g, g, g, indexing="ij")
        lut, graded, after = np.stack([r.ravel(), gg.ravel(), b.ravel()], 1), render, before
    write_cube(lut, out / "grade.cube", f"refmatch {label}")
    Image.fromarray((graded * 255).round().astype(np.uint8)).save(out / "graded.png")
    return {"de_before": round(before, 2), "de_after_lut": round(after, 2), "cube": str(out / "grade.cube"),
            "skipped": skipped}


# ---------- depth ----------

def run_depth(images: list[Path], out: Path, notes: list, device: str = "auto") -> dict:
    """Depth maps for the images in one model load, in this Python (card if it can see one, else the processor).
    Skipped with a note if the model cannot run (no card for --device cuda, no model copy and no network)."""
    cmd = [sys.executable, str(HERE / "refmatch_depth.py"), "--device", device, str(out)] + [str(p) for p in images]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        notes.append("depth timed out; skipped")
        return {}
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    if r.returncode == 0 and line:
        res = json.loads(line[7:])
        if device == "auto" and res.get("device") == "cpu":
            notes.append("no NVIDIA card was usable from this Python, so depth ran on the processor (slower)")
        return res
    notes.append(f"depth failed: {(r.stderr.strip().splitlines() or ['no output'])[-1][:200]}")
    notes.append("depth skipped: the depth model could not run")
    return {}


def depth_corr(a_npy: Path, b_npy: Path, w: int, h: int) -> float | None:
    if not (a_npy.is_file() and b_npy.is_file()):
        return None
    a = cv2.resize(np.load(a_npy), (w, h), interpolation=cv2.INTER_AREA).ravel()
    b = cv2.resize(np.load(b_npy), (w, h), interpolation=cv2.INTER_AREA).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(spearmanr(a[::4], b[::4]).statistic)


# ---------- bars and scoring ----------

def load_bars() -> dict:
    return json.loads(BARS_FILE.read_text(encoding="utf-8"))


def save_bars(cfg: dict) -> None:
    tmp = BARS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, BARS_FILE)


def combined_score(norm: dict, weights: dict) -> float | None:
    """Weighted mean x 100 over the scores present (the one formula; the calibration refit uses it too)."""
    w = {k: v for k, v in weights.items() if v > 0 and norm.get(k) is not None}
    return round(100 * sum(norm[k] * v for k, v in w.items()) / sum(w.values()), 1) if w else None


def judge(norm: dict, bar: dict) -> tuple[float | None, int | None, list]:
    """Combined score, pass (1/0, None when a required score is missing) and the reasons it failed."""
    combined = combined_score(norm, bar["weights"])
    missing = [k for k in bar["min"] if norm.get(k) is None]
    fails = [f"{k} {norm[k]:.2f} under {v}" for k, v in bar["min"].items() if norm.get(k) is not None and norm[k] < v]
    if combined is not None and combined < bar["combined_min"]:
        fails.append(f"combined {combined} under {bar['combined_min']}")
    if missing:
        return combined, None, fails + [f"missing {', '.join(missing)}"]
    return combined, int(not fails), fails


def measure(ref: np.ndarray, ren: np.ndarray, notes: list) -> tuple[dict, dict, dict]:
    lab_r, lab_s = rgb2lab(ref), rgb2lab(ren)
    mr, ms = silhouette(lab_r), silhouette(lab_s, hint=background_lab(lab_r))
    er, es = edges(lab_r), edges(lab_s)
    de = deltaE_ciede2000(lab_r, lab_s)
    blur = lambda L: cv2.GaussianBlur(L.astype(np.float32), (0, 0), 1.5)
    ssim = float(structural_similarity(blur(lab_r[..., 0]), blur(lab_s[..., 0]), data_range=100))
    lp = lpips_distance(ref, ren, notes)
    union = (mr | ms).sum()
    raw = {"iou": round(float((mr & ms).sum() / union), 3) if union else None,
           "edge_f1": None if (f := edge_f1(er, es)) is None else round(f, 3),
           "ssim": round(ssim, 3), "de_mean": round(float(de.mean()), 2),
           "lpips": None if lp is None else round(lp, 3)}
    norm = {"silhouette": raw["iou"], "edges": raw["edge_f1"], "structure": max(0.0, raw["ssim"]),
            "colour": round(max(0.0, 1 - raw["de_mean"] / DE_SCALE), 3),
            "perceptual": None if lp is None else round(max(0.0, 1 - lp), 3), "depth": None}
    return raw, norm, {"mr": mr, "ms": ms, "er": er, "es": es, "de": de}


def _u8(a):
    return (np.clip(a, 0, 1) * 255).round().astype(np.uint8)


def side_by_side(ref, ren, maps, pal, out: Path) -> Path:
    h, w = ref.shape[:2]
    heat = cv2.applyColorMap(_u8(np.clip(maps["de"] / DE_SCALE, 0, 1)), cv2.COLORMAP_INFERNO)[..., ::-1] / 255
    sil = np.zeros_like(ref)
    sil[..., 0] = maps["mr"]
    sil[..., 1] = maps["ms"]
    edg = np.zeros_like(ref)
    edg[..., 0] = maps["er"]
    edg[..., 1] = maps["es"]
    panels = [ref, ren, heat, sil * 0.85 + 0.1, edg * 0.9 + 0.05]
    titles = ["reference", "render", "colour difference", "silhouette (red ref, green render)", "edges"]
    gap, top, strip = 8, 22, 28
    canvas = Image.new("RGB", (len(panels) * (w + gap) - gap, top + h + strip), (246, 244, 239))
    d = ImageDraw.Draw(canvas)
    for i, (p, t) in enumerate(zip(panels, titles)):
        x = i * (w + gap)
        canvas.paste(Image.fromarray(_u8(p)), (x, top))
        d.text((x + 2, 5), t, fill=(40, 40, 40))
    x = 0
    for c in pal:
        cw = round(c["share"] * canvas.width)
        d.rectangle([x, top + h + 4, x + cw, top + h + strip - 2], fill=c["hex"])
        x += cw
    path = out / "side-by-side.png"
    canvas.save(path)
    return path


def write_report(out: Path, res: dict) -> Path:
    e = html.escape
    rows = "".join(f"<tr><td>{e(k)}</td><td>{'-' if v is None else v}</td><td>{e(str(res['bar']['min'].get(k, '')))}</td>"
                   f"<td>{res['bar']['weights'].get(k, 0)}</td></tr>" for k, v in res["norm"].items())
    sw = "".join(f'<span class="sw" style="background:{c["hex"]}"></span><code>{c["hex"]} {c["share"]:.0%}</code> '
                 for c in res["palette"])
    verdict = {1: "PASS", 0: "FAIL", None: "INCOMPLETE"}[res["passed"]]
    notes = "".join(f"<li>{e(n)}</li>" for n in res["notes"] + res["fails"]) or "<li>none</li>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reference match</title><style>
body{{font:15px/1.5 system-ui,sans-serif;background:#f6f4ef;color:#222;margin:0;padding:16px}}
img{{max-width:100%;height:auto;border-radius:6px}} table{{border-collapse:collapse}} td,th{{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}}
.sw{{display:inline-block;width:18px;height:18px;border-radius:4px;vertical-align:middle;margin-right:4px}}
.v{{font-size:22px;font-weight:600}}</style></head><body>
<h1>Reference match: {e(res['label'])}</h1>
<p class="v">{verdict} &middot; combined {res['combined']} (bar {res['bar']['combined_min']}, kind {e(res['kind'])})</p>
<p><img src="side-by-side.png" alt="reference, render, colour difference, silhouette and edge maps side by side"></p>
<h2>Scores (0 to 1, higher is closer)</h2><table><tr><th>score</th><th>value</th><th>minimum</th><th>weight</th></tr>{rows}</table>
<p>Raw: {e(json.dumps(res['raw']))}</p>
<h2>Reference palette</h2><p>{sw}</p>
<h2>Colour by region of the reference (mean CIEDE2000)</h2><p>{e(json.dumps(res['de_regions']))}</p>
<h2>Grade</h2><p>{"No grade helps: the render is already as close in colour as a global grade can bring it; grade.cube is an identity table." if res['grade'].get('skipped') else f"grade.cube moves the render's mean CIEDE2000 from {res['grade'].get('de_before')} to {res['grade'].get('de_after_lut')}; graded.png shows it."}</p>
<h2>Notes and reasons</h2><ul>{notes}</ul>
<p>Match #{res.get('match_id', '-')} &middot; {e(res['created_at'])} &middot; reference {e(res['reference'])} &middot; render {e(res['render'])}</p>
</body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


def _record(res: dict) -> int:
    with refmatch_store.session(BARS_FILE) as con:
        cur = con.execute(
            "INSERT INTO reference_matches (created_at, kind, build_id, label, reference, render, report_dir, combined,"
            " metrics, bar, passed, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (res["created_at"], res["kind"], res["build"], res["label"], res["reference"], res["render"], res["out"],
             res["combined"], json.dumps({"raw": res["raw"], "norm": res["norm"], "de_regions": res["de_regions"]}),
             json.dumps(res["bar"]), res["passed"], json.dumps(res["notes"] + res["fails"])))
        return cur.lastrowid


def run(ref_path, render_paths, kind: str, build: int | None = None, label: str | None = None, depth: bool = True,
        out_root: Path | None = None, record: bool = True, device: str = "auto") -> list[dict]:
    """Score each render against the reference; one report folder per render. Returns the results, best first."""
    cfg = load_bars()
    if kind not in cfg["kinds"]:
        raise RefmatchError(f"unknown job kind {kind!r}; known: {', '.join(cfg['kinds'])}")
    bar = cfg["kinds"][kind]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]+", "-", label or Path(ref_path).stem).strip("-.")[:60] or "run"
    base = Path(out_root or os.environ.get("REFMATCH_OUT") or Path.cwd() / "refmatch-output") / f"{stamp}-{safe}"
    base.mkdir(parents=True, exist_ok=True)
    ref_notes: list = []
    ref_full = load_rgb(ref_path, ref_notes)
    w, h = work_size(ref_full)
    ref = resize(ref_full, w, h)
    lab_ref = rgb2lab(ref)
    pal, labels = palette(lab_ref)
    bg = lab2rgb(background_lab(lab_ref)[None, None])[0, 0]
    (base / "palette.json").write_text(json.dumps(pal, indent=1), encoding="utf-8")

    renders = []
    for i, rp in enumerate(render_paths):
        notes = list(ref_notes)
        name = f"{i + 1:02d}-{Path(rp).stem}"
        out = base / name
        out.mkdir(exist_ok=True)
        ren_full = load_rgb(rp, notes, bg=bg)
        ren = fit_to(ren_full, w, h, notes, Path(rp).name)
        clar = clarity(ref_full, ren_full)
        del ren_full
        Image.fromarray(_u8(ren)).save(out / f"{name}-fit.png")
        renders.append((rp, name, out, ren, notes, clar))
    Image.fromarray(_u8(ref)).save(base / "reference-fit.png")

    depth_res = {}
    if depth:
        dnotes: list = []
        depth_res = run_depth([base / "reference-fit.png"] + [o / f"{n}-fit.png" for _, n, o, *_ in renders],
                              base / "depth", dnotes, device)
        for r in renders:
            r[4].extend(dnotes)

    results = []
    for rp, name, out, ren, notes, (clar_raw, clar) in renders:
        raw, norm, maps = measure(ref, ren, notes)
        raw["clarity"] = clar_raw
        norm["clarity"] = clar
        dc = None
        if depth_res:
            dc = depth_corr(base / "depth" / "reference-fit.depth.npy", base / "depth" / f"{name}-fit.depth.npy", w, h)
        raw["depth_corr"] = None if dc is None else round(dc, 3)
        norm["depth"] = None if dc is None else round((dc + 1) / 2, 3)
        combined, passed, fails = judge(norm, bar)
        regions = {}
        for j, c in enumerate(pal):
            m = labels == j
            if m.any():
                regions[c["hex"]] = {"share": c["share"], "de": round(float(maps["de"][m].mean()), 2)}
        res = {"created_at": datetime.now().isoformat(timespec="seconds"), "kind": kind, "build": build,
               "label": f"{label or Path(ref_path).stem} / {Path(rp).name}", "reference": str(Path(ref_path).resolve()),
               "render": str(Path(rp).resolve()), "out": str(out), "palette": pal, "raw": raw, "norm": norm,
               "de_regions": regions, "bar": bar, "combined": combined, "passed": passed, "fails": fails,
               "notes": notes, "grade": grade(ref, ren, out, name), "depth": depth_res}
        side_by_side(ref, ren, maps, pal, out)
        if record:
            res["match_id"] = _record(res)
        write_report(out, res)
        (out / "report.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        results.append(res)
    return sorted(results, key=lambda r: -(r["combined"] or 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reference matcher: measure an inspiration image and score a recreation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ref", required=True)
    r.add_argument("--render", required=True, action="append")
    r.add_argument("--kind", required=True)
    r.add_argument("--build", type=int, help="optional number of your own to file the match under")
    r.add_argument("--label")
    r.add_argument("--no-depth", action="store_true")
    r.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="where the depth model runs")
    r.add_argument("--no-record", action="store_true")
    r.add_argument("--out")
    p = sub.add_parser("palette")
    p.add_argument("--ref", required=True)
    p.add_argument("--k", type=int, default=6)
    g = sub.add_parser("grade")
    g.add_argument("--ref", required=True)
    g.add_argument("--render", required=True)
    g.add_argument("--out", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("match_id", type=int)
    v.add_argument("decision", choices=["accept", "reject"])
    v.add_argument("--words", default="")
    v.add_argument("--seen", default="seen", choices=["blind", "seen", "seen-possibly"],
                   help="blind: you judged before seeing the score (counts most when tuning)")
    a = sub.add_parser("agreement")
    a.add_argument("--kind")
    b = sub.add_parser("bars")
    b.add_argument("--kind")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            res = run(args.ref, args.render, args.kind, args.build, args.label, not args.no_depth,
                      Path(args.out) if args.out else None, not args.no_record, args.device)
            for x in res:
                word = {1: "PASS", 0: "FAIL", None: "INCOMPLETE"}[x["passed"]]
                print(f"{word:10} {x['combined']!s:>5}  match #{x.get('match_id', '-')}  {x['label']}")
                print(f"           {json.dumps(x['raw'])}")
                for n in x["notes"] + x["fails"]:
                    print(f"           - {n}")
                print(f"           report: {Path(x['out']) / 'report.html'}")
        elif args.cmd == "palette":
            rgb = load_rgb(args.ref, [])
            print(json.dumps(palette(rgb2lab(resize(rgb, *work_size(rgb))), args.k)[0], indent=1))
        elif args.cmd == "grade":
            notes: list = []
            ref = load_rgb(args.ref, notes)
            w, h = work_size(ref)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            ren = fit_to(load_rgb(args.render, notes), w, h, notes, Path(args.render).name)
            print(json.dumps(grade(resize(ref, w, h), ren, out, Path(args.render).stem) | {"notes": notes}))
        elif args.cmd == "verdict":
            import refmatch_calibrate
            print(json.dumps(refmatch_calibrate.verdict(args.match_id, args.decision, args.words, args.seen)))
        elif args.cmd == "agreement":
            import refmatch_calibrate
            print(json.dumps(refmatch_calibrate.agreement(args.kind), indent=1))
        elif args.cmd == "bars":
            kinds = load_bars()["kinds"]
            print(json.dumps(kinds[args.kind] if args.kind else kinds, indent=1))
    except RefmatchError as e:
        print(f"refmatch: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    # refmatch_calibrate imports refmatch by name; without this it loads a second copy whose RefmatchError main() cannot catch
    sys.modules.setdefault("refmatch", sys.modules[__name__])
    sys.exit(main())

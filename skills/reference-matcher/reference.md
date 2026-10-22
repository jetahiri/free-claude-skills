# Reference matcher: details

Read this when you need the exact meaning of a score, the list of job kinds, how verdicts move the bars, or what each failure does.

## Job kinds

Bars and weights live in `scripts/refmatch_bars.json`; `python SKILL_DIR/scripts/refmatch.py bars` prints them.

| Kind | Use it for |
|---|---|
| `3d-background` | a 3D scene or background recreated from an inspiration image |
| `3d-clay` | the plain grey shape check before materials: shape scores only |
| `web-page` | a page screenshot against a design reference |
| `video-frame` | a video frame against a reference frame |

## Scores (0 to 1, higher is closer)

| Score | How |
|---|---|
| silhouette | Overlap of forms-versus-background masks: CIEDE2000 distance from the border background colour, Otsu split, closed and hole-filled; the render's background is the border colour nearest the reference's |
| edges | F1 of Canny edge maps within 2 px |
| structure | SSIM of lightness after a light blur |
| colour | 1 - mean CIEDE2000 / 25 (0.8 means a mean difference of 5); per palette region in the report |
| perceptual | 1 - LPIPS (AlexNet, on the processor) |
| depth | Spearman correlation of Depth Anything V2 Small depth maps, mapped to 0..1 (card when torch can see one, else the processor) |
| clarity | Mean of three closeness ratios at near-full size: noise in flat areas, Laplacian sharpness on the main edges, fine-grain energy |

Combined = weighted mean x 100. Pass = every minimum holds and combined reaches the bar. A missing required score makes the result INCOMPLETE, never a pass.

## Output of a run

Each render gets a folder `<output>/<time>-<label>/<nn>-<render name>/` with `report.html` (one page), `side-by-side.png` (reference, render, colour difference heat map, silhouettes, edges, palette strip), `grade.cube` (33-point LUT that moves the render toward the reference; loads in video editors and web LUT tools; an identity table when no global grade would bring the render closer), `graded.png`, `report.json`, and depth maps in `depth/` (one level up). The match number is printed and stored in the records file.

Other commands: `palette --ref R` (main colours in Lab and their share), `grade --ref R --render S --out DIR`, `bars [--kind K]`, `agreement [--kind K]`.

## Verdicts tune the bars (your eye wins)

```
python SKILL_DIR/scripts/refmatch.py verdict <match #> accept|reject --words "<your words>" --seen blind|seen|seen-possibly
```

`blind` means you judged before seeing the score (no anchoring) and counts most. After every verdict the kind's combined bar moves the least it can so that every verdict on record agrees with it. An accept also lowers any per-score minimum that run missed. When no bar can agree with every verdict (an accepted run scored below a rejected one), the bar goes to the cut that agrees with the most, blind verdicts weighing most, and the conflict is logged. Every verdict is then replayed against the bars in force, and any the tool would still judge the other way is logged as a clash. A second verdict on the same run replaces the first, and the first stays in the run's notes. With 8 or more verdicts of both kinds, the weights are refit by logistic regression, using the same combined formula as the pass check. `agreement` reports how often the tool's answer (as it stood when it scored) matched your verdict, per kind and per week.

## Failure handling

Missing or unreadable image, or under 32 px a side: stops with one plain line. Over 80 million pixels: refused from the file header, before decoding. Longer than 8:1: refused. Phone photos stored sideways are turned upright. A label with characters a folder name cannot hold is cleaned. Transparent PNG: flattened over the reference's background colour, with a note. Different aspect ratio: centre-cropped, never stretched, with a note. No usable NVIDIA card: depth runs on the processor, with a note. `--device cuda` with no card: depth is skipped with one plain line. Depth model cannot load: depth skipped, with a note, and the run still completes. Perceptual model cannot load: perceptual is missing, so the result is INCOMPLETE.

## Known limits

- The silhouette is colour-based. On softly lit forms whose shadows match the background it can under-read shape; the `3d-clay` bar was lowered by a verdict to account for that.
- The first run downloads the Depth Anything V2 Small weights from Hugging Face and the AlexNet weights (about 233 MB) from download.pytorch.org. After that no network is used.

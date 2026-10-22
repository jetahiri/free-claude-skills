# Inverse fit

Makes a preset match its reference photo automatically, scored by the reference-matcher skill. Two steps, both reusable for any image. The scoring comes from `../reference-matcher/scripts` (set `REFMATCH_DIR` if the skills are not side by side).

## 1. Normals and refined depth (optional)

```
python scripts/normals_from_ref.py --image depth/reference.jpg --depth depth/depth.npy --mask depth/mask.npy --out fit/
```

Needs `diffusers` and `psutil` as well as the packages in `requirements.txt`, and downloads the Marigold normals checkpoint (CreativeML Open RAIL++-M, see LICENCES.md) on first use. Measured on a 1200 x 2141 image: 3,260 MB of card memory and 5,906 MB of main memory while the model loads. Re-running only the depth refinement (`--reuse --weight W --name depth-refined-W`) needs 180 MB of card memory and 1,140 MB of main memory. Lower `--weight` lets the photo's surface directions reshape the depth more.

## 2. Fit

```
python scripts/inverse_fit.py fit --settings <preset.json> --ref <reference> --out fit/run1 --depths fit/depth-refined.npy,fit/depth-refined-0.005.npy --scale 0.33 --samples 32 --pop 6 --max-gens 6
```

One unattended job: renders each candidate with `image_to_3d.py` on a half-size grid (about 5 s each on an NVIDIA card), fits the finish (colour table, sharpening, grain) in numpy, scores with refmatch's own functions (no depth score, so it matches a refmatch run without depth), and writes:

- `log.jsonl`: every candidate, its settings and scores
- `summary.json`: start score, best per stage, the winning settings
- `best1..3-*.png`: the three best finished renders
- `<preset>.fit.json` beside the original preset: same format plus `finish` and `fit` blocks (image_to_3d.py ignores both)

Stages: which depth map, then shape (smooth, curl, curl width, thickness, gap depth), then material and light (roughness, subsurface, sheen, three light powers, world strength, f-stop). Each stage stops after 2 generations without a 0.3 point gain.

After the fit, render the fitted preset full size, then `inverse_fit.py finish --settings <fit.json> --ref <ref> --render <png> --out <finished png> --refit` and score it with `refmatch.py run`.

Why Blender in the loop instead of Mitsuba 3 or a PyTorch renderer: the result must look the same in Cycles; a second renderer would fit its own subsurface, colour view and depth of field, and the values would not carry over.

## Notes

- Fitting an already fitted preset (`<name>.fit.json`) rewrites that same file; copy it first to keep the earlier fit.
- If three renders fail in a row the run stops, but still writes the summary (with `aborted`), the best renders found so far and the fitted preset; each failed render leaves an `.err.txt` file beside it.
- Without an NVIDIA card the fit still runs, on the processor, and takes far longer per candidate; lower `--samples`, `--scale` and `--max-gens`.

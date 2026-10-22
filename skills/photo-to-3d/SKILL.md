---
name: photo-to-3d
description: "Turn a photo into a lit 3D scene in Blender, then fit its material and light to match the photo. Use when asked: photo to 3D, turn this image into a 3D scene, rebuild this picture in Blender, make a 3D version of this photo, recreate this background in 3D, fit the 3D render to the reference."
---

# Photo to 3D

## Trigger

Fires on: "photo to 3D", "turn this image into a 3D scene", "rebuild this picture in Blender", "make a 3D version of this photo", "recreate this background in 3D", "fit the render to the reference".

If the person already has a render and only wants it scored, the `reference-matcher` skill wins. This skill calls `reference-matcher` for its scores, so both folders must be installed side by side.

## Desired outcome

A PNG render of the photo's forms as a lit 3D scene exists at the size asked for, made from a depth map of the photo, with a refmatch score printed against the photo, and a preset file that reproduces it.

## Procedure

`SKILL_DIR` is the folder that holds this file (see the repository README). `WORK` is a new folder you make for this photo; every file below lands inside it.

1. Copy the photo into `WORK`. Make the depth map and silhouette (an NVIDIA card is used when torch can see one):

   ```
   python SKILL_DIR/scripts/depth_from_ref.py WORK/photo.jpg WORK/depth
   ```

   The printed RESULT line gives the image size. The photo needs a fairly plain background in its left corners, because the silhouette is the colour distance from them.

2. Copy `SKILL_DIR/presets/starter.json` to `WORK/preset.json`. Set `image.image` to your photo, and `image.depth` and `image.mask` to the `depth.npy` and `mask.npy` that step 1 wrote (all paths relative to the preset), `image.width` and `image.height` to the printed size, and `geometry.grid` to half of that size.
3. Grey shape check first. Run Blender in the background (`BLENDER` is the full path of `blender.exe` when it is not on the PATH), then score the shape:

   ```
   blender -b --factory-startup --python-exit-code 1 --python SKILL_DIR/scripts/image_to_3d.py -- --settings WORK/preset.json --stage clay --out WORK/clay.png --scale 0.5 --samples 32
   python SKILL_DIR/../reference-matcher/scripts/refmatch.py run --ref WORK/photo.jpg --render WORK/clay.png --kind 3d-clay --no-depth
   ```

   If the shape is off, change `geometry.*` in the preset using `SKILL_DIR/docs/SETTINGS.md` and repeat; stop after three rounds and report the best score.
4. Optional closed mesh (needs an NVIDIA card and a TripoSG install, see Failure branches): `python SKILL_DIR/scripts/image_to_mesh.py WORK/photo.jpg WORK/depth/mask.npy WORK/form.glb --engine <TripoSG folder>`, add the `mesh` block from `docs/SETTINGS.md` to the preset, then `python SKILL_DIR/scripts/place_mesh.py WORK/preset.json --write`.
5. Optional automatic fit of material and light (long, unattended; steps and stages in `SKILL_DIR/docs/INVERSE-FIT.md`):

   ```
   python SKILL_DIR/scripts/inverse_fit.py fit --settings WORK/preset.json --ref WORK/photo.jpg --out WORK/fit
   ```

   It writes `WORK/preset.fit.json`; use that file as `--settings` in the next step.
6. Final render and score:

   ```
   blender -b --factory-startup --python-exit-code 1 --python SKILL_DIR/scripts/image_to_3d.py -- --settings WORK/preset.json (or WORK/preset.fit.json) --stage final --out WORK/final.png
   python SKILL_DIR/../reference-matcher/scripts/refmatch.py run --ref WORK/photo.jpg --render WORK/final.png --kind 3d-background
   ```

7. Tell the person the score, the lowest measure and what the render looks like next to the photo, and where `WORK/final.png` and the preset are.

Cost: every step is code or a local model; Claude only decides which setting to change next. A render without an NVIDIA card works but is much slower; lower `--scale` and `--samples` for drafts.

## Stop test

Done when `WORK/final.png` exists at the preset's size, the refmatch run printed a combined score for it, and the person has been told both. For a shape-only request, done when `WORK/clay.png` exists and its 3d-clay score has been reported.

## Failure branches

- `depth_from_ref.py` prints "cannot read image": the file is not a picture or is damaged; ask for another file. First run downloads the depth model: on a network error retry once, then stop and say the model could not be downloaded.
- Blender prints "Blender not found" (from `inverse_fit.py`) or the command is not recognised: install Blender from blender.org (tested with 5.1.0) or set `BLENDER` to its full path, then retry once; if it still fails, stop.
- Blender exits non-zero: read the last lines of its output. "missing depth file" means step 1 or the preset paths are wrong: fix and retry once. Any other error: retry once, then stop and show the lines.
- Blender prints "NOTICE no NVIDIA card found": the render is running on the processor. Continue, tell the person it will be slow, and lower `--scale` and `--samples`.
- `image_to_mesh.py` prints "TripoSG not found", "TripoSG weights missing" or "needs an NVIDIA card": the mesh is optional. Skip step 4 (the depth surface works without it) or ask the person to follow the TripoSG install steps at https://github.com/VAST-AI-Research/TripoSG and download only the `VAST-AI/TripoSG` weights; never download or load RMBG-1.4 (non-commercial).
- `normals_from_ref.py` (the optional normals step) reports a missing `diffusers`: `pip install diffusers psutil` once, else skip it.
- `inverse_fit.py` aborts after three failed renders: it still writes a summary and the best result found; read the `.err.txt` files, fix the preset, and run once more; if it aborts again, stop.
- `inverse_fit.py` prints "needs the reference-matcher skill": install it beside this folder or set `REFMATCH_DIR`.

## Writes

- Everything goes inside `WORK` or the `--out` folder you pass: depth maps, meshes, renders, fit logs, fitted presets.
- Model weights are cached by Hugging Face in the user's cache folder on first download.
- Scores are recorded by `reference-matcher` as described in its own SKILL.md.
- No data leaves the machine.

## When NOT to use

- Scoring an existing render: use `reference-matcher`.
- A 3D object with no photo to copy: model it directly in Blender.
- Photo editing, upscaling or colour grading of the photo itself: this skill only builds a 3D scene from it; for a colour grade use `reference-matcher`'s `grade` command.
- Animated loops are possible (`--stage anim`) but are not covered by the stop test above; read `docs/SETTINGS.md` first.

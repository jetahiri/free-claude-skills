# Free skills for Claude Code

Two skills that run on your own machine and need no extra account, key or paid service beyond Claude Code itself.

| Skill | What it does for you |
|---|---|
| `reference-matcher` | Scores how close your recreation is to an inspiration image, in numbers instead of by eye: shape, edges, structure, colour, a perceptual score, depth and sharpness. It also writes a colour grade file that moves your render toward the reference, and learns from your own accept or reject verdicts. |
| `photo-to-3d` | Turns a photo into a lit 3D scene in Blender: depth and silhouette from the photo, an optional closed mesh, then an automatic search for the material and light settings that score closest to the photo. |

## Install

1. Copy the two folders inside `skills/` into your Claude Code skills folder: `~/.claude/skills/` for every project, or `.claude/skills/` inside one project. Keep them side by side, because `photo-to-3d` finds `reference-matcher` as its sibling folder.
2. Install the Python packages: `pip install -r requirements.txt`. For an NVIDIA card, install the CUDA build of torch first (https://pytorch.org/get-started/locally/); the default Windows build runs on the processor only.
3. For `photo-to-3d` install Blender (https://www.blender.org) and either put it on your PATH or set the `BLENDER` environment variable to the full path of `blender.exe`.
4. Ask Claude Code for what the skill does ("score my render against this reference", "turn this photo into a 3D scene"); each `SKILL.md` lists its trigger phrases.

## What SKILL_DIR means

Every command in a `SKILL.md` is written as `python SKILL_DIR/scripts/...`. `SKILL_DIR` stands for the folder that holds that `SKILL.md`, for example `~/.claude/skills/reference-matcher`. Claude substitutes the real path when it runs the command; if you run a command yourself, substitute it by hand.

## What runs where

- Everything runs locally. Nothing is uploaded and no key is needed.
- Models are downloaded once, on first use, from Hugging Face (depth, optional normals, optional mesh) and from download.pytorch.org (the AlexNet weights behind the perceptual score). The `LICENCES.md` table lists each one with its licence.
- An NVIDIA card is used when torch and Blender can see one and is not required, except for the optional mesh step (TripoSG), which needs one. Without a card the depth model runs on the processor (slower), and Blender renders on the processor (much slower).
- Results go to your own folders: `reference-matcher` writes report folders to `./refmatch-output/` (or `REFMATCH_OUT`) and keeps its record in `refmatch_records.sqlite` beside its bars file (or `REFMATCH_DB`); `photo-to-3d` writes where you point `--out`.

## Tested with

Windows 11, Python 3.12, Blender 5.1.0, numpy 2.5, torch 2.14 (processor build) and the other versions listed in `LICENCES.md`. Other versions and other systems are untested. The glare step in a `photo-to-3d` preset uses the Blender 5 compositor, so leave the `glare` block out of a preset on an older Blender.

## Run the tests

```
python -m pytest skills tests -q
```

The tests run on the processor with generated images, and none downloads a model. Tests that load a model (the depth model and the perceptual score) run only with `REFMATCH_TEST_MODELS=1`; Blender tests skip when Blender is not found.

## When something goes wrong

| What you see | What it means and what to do |
|---|---|
| "no NVIDIA card was usable from this Python" | Depth ran on the processor. Install the CUDA build of torch for speed, or ignore the note. |
| "--device cuda was asked for, but this Python's torch cannot see an NVIDIA card" | Use `--device cpu` or `--device auto`. |
| "depth skipped: the depth model could not run" | The model could not be downloaded or loaded (no network, or a full disk). The run still finishes without the depth score; connect once and run again. |
| "perceptual score unavailable" | The AlexNet weights could not be downloaded. The result is INCOMPLETE, never a pass; connect once and run again. |
| "Blender not found" | Install Blender or set `BLENDER` to the full path of its program file. |
| "cannot read image" or "image too small" | The file is not a readable picture, or is under 32 pixels a side. Use another file. |
| "TripoSG not found" or "TripoSG weights missing" | The mesh step is optional. Skip it, or follow the steps in `skills/photo-to-3d/SKILL.md`. |

## Licences

Our code is MIT (`LICENSE`), except `image_to_3d.py`, which runs inside Blender and is GPL-3.0-or-later. `LICENCES.md` has one row per model and library, and names the models left out on purpose because their licences are non-commercial.

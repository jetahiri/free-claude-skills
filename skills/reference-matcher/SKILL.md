---
name: reference-matcher
description: "Score a recreation against its inspiration image in numbers, not by eye. Use when asked: score my render against this reference, how close is my recreation to this image, compare my design to the reference, reference match, measure how faithful this copy is, make a colour grade from a reference, record my verdict on this recreation."
---

# Reference matcher

## Trigger

Fires on: "score my render against this reference", "how close is my recreation to this image", "compare my design to the reference", "reference match", "make a colour grade from this reference", "accept" or "reject" a scored recreation.

If a request is "turn this photo into a 3D scene", the `photo-to-3d` skill fires first and calls this one to score its result. When the person already has a render and wants a number, this skill wins.

## Desired outcome

One report folder per render exists, each holding a pass, fail or incomplete verdict with the lowest-scoring measure named, a side-by-side picture, and a colour grade file, and the person has been told in plain words what to change next.

## Procedure

`SKILL_DIR` is the folder that holds this file (see the repository README).

1. Confirm the reference image and each render exist and are picture files. Choose the job kind: `3d-background`, `3d-clay`, `web-page` or `video-frame` (what each is for: `SKILL_DIR/reference.md`).
2. Run the score. Add one `--render` per candidate; add `--no-depth` for a faster run without the depth score, `--device cpu` to keep depth off the card, `--label` for a short name.

   ```
   python SKILL_DIR/scripts/refmatch.py run --ref <reference image> --render <render> --kind <kind> --label <short name>
   ```

3. Read the printed lines: PASS, FAIL or INCOMPLETE, the combined number, the raw measures, each note, and the report path. Open `report.html` and `side-by-side.png` in the report folder.
4. Tell the person the verdict, the lowest-scoring measure and the concrete change it points to (shape and silhouette: move or resize the forms; colour: use `grade.cube` or change the palette; clarity: less noise or more sharpness). When several renders were scored, name the best first.
5. If the person gives a verdict on a run, record it with the number printed in step 3, using their own words, and say whether the bar moved:

   ```
   python SKILL_DIR/scripts/refmatch.py verdict <match number> accept|reject --words "<their words>" --seen blind|seen|seen-possibly
   ```

6. Other commands when asked: `palette --ref <image>` (main colours), `grade --ref <image> --render <image> --out <folder>` (grade only), `bars` (pass bars in force), `agreement` (how often the score matched the person's verdicts).

Cost: every measurement is plain code. Two small local models are used (depth and the perceptual score) and no cloud call is made.

## Stop test

Done when the run exited with code 0, every render's folder contains `report.html`, `side-by-side.png`, `grade.cube`, `graded.png` and `report.json`, and the person has been told the verdict. A verdict request is done when the command printed its JSON with the match number.

## Failure branches

- `refmatch.py run` exits with code 2 and one line (image missing, unreadable, under 32 px, too large, wrong shape, unknown kind): fix that input, run once more; if it stops again, show the line to the person and stop.
- ImportError for a package: run `pip install -r requirements.txt` from the repository folder once, then retry once; if it fails again, show the error and stop.
- Depth note "depth failed" or "depth skipped": the run still finishes without the depth score; report it as a note. One retry only if the cause was a network drop during the model download.
- Note "perceptual score unavailable": the result is INCOMPLETE, never a pass. Retry once with a network connection (the AlexNet weights download on first use); if it persists, report INCOMPLETE as it is.
- `--device cuda` with no usable card: use `--device auto` or `--device cpu` instead of retrying.
- `verdict` with an unknown match number: the number is printed after each scored run ("match #N"); ask the person which run is meant.

## Writes

- Report folders under `./refmatch-output/` in the current folder (or the folder named by `--out` or the `REFMATCH_OUT` variable).
- `refmatch_records.sqlite` beside `SKILL_DIR/scripts/refmatch_bars.json` (or the file named by `REFMATCH_DB`), created on first use: every scored run, every verdict, every bar change with its reason.
- `refmatch_bars.json` itself, only when a verdict moves a bar.
- Nothing else is written, and no result data is sent anywhere (only the one-time model downloads use the network).

## When NOT to use

- Judging whether a design is good or on-brand: this measures closeness to one reference only, so use a human or a design review.
- Making the 3D scene itself: use the `photo-to-3d` skill, then come back here to score it.
- Comparing two of your own renders with each other: score both against the same reference and compare the numbers.

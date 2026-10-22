# Image to 3D: the settings file

`scripts/image_to_3d.py` turns a reference image into a lit, procedural 3D scene. Everything about one image lives in a settings file (a named preset, JSON). A new image needs a new settings file and its inputs; the scripts stay as they are. `presets/starter.json` is a complete one to copy.

## Steps for a new image
1. Depth and silhouette: `depth_from_ref.py <image> <out folder>` (Depth Anything V2 Small, Apache 2.0) writes `depth.npy` and `mask.npy` and prints the image size. A soft silhouette (`alpha.npy`, from a per-row background colour model) suits blurred photos better; point `image.mask` at it if you make one.
2. A closed mesh (optional, needs an NVIDIA card): `image_to_mesh.py <image> <alpha.npy> <out.glb> --engine <TripoSG folder>` (TripoSG, MIT; never loads its non-commercial background remover). Add a `mesh` block to the preset (below).
3. Placement of the mesh on the silhouette, no rendering: `place_mesh.py <preset> --write`.
4. Material and light fit against the image: `inverse_fit.py fit --settings <preset> --ref <image> --out <folder>`, then `inverse_fit.py finish ... --refit`, then the reference-matcher skill's `refmatch.py run ...` for the official score. Details: `INVERSE-FIT.md`.
5. Render: `blender -b --factory-startup --python-exit-code 1 --python image_to_3d.py -- --settings <preset> --stage clay|final|anim|projection --out ...`.

## What each setting changes
| Setting | Changes |
|---|---|
| `image.*` | Which image, depth map and silhouette; the frame size |
| `geometry.grid`, `smooth` | Detail of the depth surface; higher smooth = softer, rounder folds |
| `geometry.thickness`, `near` | How deep the relief is: more thickness = deeper folds |
| `geometry.curl`, `curl_width` | How far each outline rolls away from the camera, and over how wide a band |
| `geometry.gap_depth` | Pushes dark gaps between folds back, separating stacked folds |
| `tubes.loops` | Optional hand-traced folds (`trace_folds.py` finds candidates): each a list of [x, y, nearness] points; `radius` = tube thickness. Edit points to reshape a fold, add a list to add one. Leave the block out for none |
| `tubes.webs`, `hide_body` | Membranes inside loops; hide the depth surface |
| `mesh.path`, `depth`, `scale`, `offset`, `yaw` | A generated closed mesh replaces the surface: where it sits, how big, how turned. Block shape: `"mesh": {"path": "mesh/form.glb", "depth": 5.6, "fit": "box", "yaw": 0, "scale": 1.0, "offset": [0.0, 0.0]}` |
| `material.*` | Colour (`base`), how far light travels inside (`subsurface_scale`, `subsurface_radius`), surface sheen and roughness, the frosted grain (`grain_strength`, `grain_scale`: about 140 gives a pebbled orange-peel relief, `grain_depth`: relief height, default 0.005) |
| `lights[]` | Each light's place (`at`), target (`aim`), size, power and colour; `warm: true` lights are left out of clay renders |
| `world`, `backdrop` | The fill colour and the backdrop gradient and warm bloom behind the forms |
| `atmosphere.*` | A thin haze between camera and backdrop: `density` (0.02 is light; 0.05 already washes the forms out), `color`, `anisotropy`, `from`/`to` (depth range). Leave the block out for no haze |
| `glare.*` | Glow on the highlights (compositor Glare, Blender 5 or newer): `type` (Bloom, Fog Glow), `threshold`, `strength`, `size`. Leave out for none |
| `camera.lens`, `fstop`, `focus_at`, `focus_distance` | Framing, how shallow the focus is, and where the sharp spot sits (on a mesh base the focus follows the ray through `focus_at` to the mesh); `focus_distance` pins the focus to a fixed distance instead |
| `motion.*` | The loop: length in frames, drift and sway amounts |
| `render.*` | Samples, size, film look |

To change the form later: swap `mesh.path` for another generated or edited mesh, or adjust `scale`, `offset` and `yaw`; the material and lights carry over unchanged.

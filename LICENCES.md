# Licences

Our code is MIT (see LICENSE), except `skills/photo-to-3d/scripts/image_to_3d.py`, which imports Blender's `bpy` and is GPL-3.0-or-later (its header says so). No model weights are included in this repository: each one is downloaded by the person running the skill, from the source in the table. Checked on 2026-10-21.

## Models the scripts load

| Model | Exact checkpoint | Licence | Commercial use | Used by | Where checked (2026-10-21) |
|---|---|---|---|---|---|
| Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 | yes | reference-matcher (depth score), photo-to-3d (`depth_from_ref.py`) | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf (card metadata: apache-2.0); https://github.com/DepthAnything/Depth-Anything-V2 (README: Small is Apache-2.0) |
| Marigold normals v1.1 | `prs-eth/marigold-normals-v1-1` | CreativeML Open RAIL++-M (the code around it is Apache-2.0) | yes, with conditions: the licence grants use with no field-of-use ban of its own, and requires that any use-based restrictions it names are passed on to people you redistribute the model to; this repository does not redistribute it | photo-to-3d (`normals_from_ref.py`, optional step) | https://huggingface.co/prs-eth/marigold-normals-v1-1 (card metadata: openrail++); https://raw.githubusercontent.com/prs-eth/Marigold/main/LICENSE-MODEL.txt (grant is perpetual, worldwide, royalty-free; Attachment A is blank in that copy); https://github.com/prs-eth/Marigold (code Apache-2.0, checkpoints RAIL++-M) |
| TripoSG | `VAST-AI/TripoSG` (weights) | MIT | yes | photo-to-3d (`image_to_mesh.py`, optional step) | https://huggingface.co/VAST-AI/TripoSG (card metadata: mit); https://github.com/VAST-AI-Research/TripoSG (code: MIT licence file) |
| LPIPS (AlexNet version) | `lpips` 0.1.4, net `alex` (linear layers ship inside the pip package) | BSD-2-Clause | yes | reference-matcher (perceptual score) | https://github.com/richzhang/PerceptualSimilarity (BSD-2-Clause); pip metadata of `lpips` 0.1.4 (BSD License) |
| AlexNet ImageNet weights | torchvision `AlexNet_Weights.IMAGENET1K_V1` (`alexnet-owt-7be5be79.pth`, downloaded from download.pytorch.org on first use) | torchvision code is BSD-3-Clause; the weights were trained on ImageNet and torchvision states that pretrained weights "may have their own licenses or terms and conditions derived from the dataset used for training" | yes for the code; the weights carry no extra restriction in the file, and the ImageNet terms are the open question for any commercial product that ships them. They are used here only to compute a similarity number, and nothing is redistributed | reference-matcher (perceptual score) | https://github.com/pytorch/vision (BSD-3-Clause); https://docs.pytorch.org/vision/stable/models.html (statement quoted in this row) |

## Excluded

| Model | Licence | Why it is not here |
|---|---|---|
| BRIA RMBG-1.4 (`briaai/RMBG-1.4`) | bria-rmbg-1.4: Creative Commons, non-commercial; commercial use needs a separate agreement with BRIA | https://huggingface.co/briaai/RMBG-1.4 (checked 2026-10-21). TripoSG's own script downloads and loads it. `image_to_mesh.py` passes its own silhouette and never loads it, and a test fails if any script downloads or loads it. Depth Anything V2 Base and Large (CC-BY-NC-4.0) are excluded for the same reason. |

## Libraries and tools

None of these is copied into the repository; `requirements.txt` names the ones pip installs.

| Name | Licence | Source checked |
|---|---|---|
| numpy 2.5.3 | BSD-3-Clause (with bundled permissive parts) | pip metadata |
| scipy 1.18.1 | BSD-3-Clause | pip metadata |
| scikit-image 0.26.0 | BSD-3-Clause | pip metadata |
| scikit-learn 1.9.1 | BSD-3-Clause | pip metadata |
| opencv-python-headless 5.0.0.93 | Apache-2.0 | pip metadata |
| Pillow 12.3.0 | MIT-CMU | pip metadata |
| torch 2.14.1 | BSD-style (Apache-2.0 and BSD parts) | pip metadata |
| torchvision 0.29.1 | BSD-3-Clause | pip metadata; https://github.com/pytorch/vision |
| transformers 5.19.0 | Apache-2.0 | pip metadata |
| huggingface_hub 1.33.0 | Apache-2.0 | pip metadata |
| diffusers (photo-to-3d normals step only) | Apache-2.0 | https://github.com/huggingface/diffusers |
| trimesh 5.1.0 | MIT | pip metadata |
| psutil 7.1.0 | BSD-3-Clause | pip metadata |
| pytest 9.1.1 (tests only) | MIT | pip metadata |
| Blender 5.1.0 (not bundled; you install it) | GPL; scripts that use `bpy` and are published must be GPL-compatible | https://www.blender.org/about/license/ |
| pymeshlab (not bundled; TripoSG's own requirements install it, and `image_to_mesh.py` imports TripoSG's mesh-simplify helper that uses it) | GPL-3.0 | https://pypi.org/project/pymeshlab/ |

Versions are the ones the tests ran against on 2026-10-21; later versions are expected to work and are untested.

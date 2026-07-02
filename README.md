# WorldFlow3D

<p align="center">
  <a href="https://princeton-computational-imaging.github.io/WorldFlow3D/">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2603.29089">
    <img src="https://img.shields.io/badge/arXiv-2603.29089-b31b1b.svg" alt="arXiv"></a>
</p>

**WorldFlow3D: Flowing Through 3D Distributions for Unbounded World Generation** (ECCV 2026).

[Amogh Joshi](https://amogh7joshi.github.io), [Julian Ost](https://ostjul.com), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

<p align="center">
  <img src="https://raw.githubusercontent.com/princeton-computational-imaging/WorldFlow3D/website/assets/images/teaser_figure_one_street-1.webp" alt="WorldFlow3D teaser" width="100%">
</p>

Diffusers-native inference for **WorldFlow3D** — hierarchical, map/layout-conditioned
3D scene generation via chunked flow matching. A self-contained inference package:
pretrained cascades load straight from the Hugging Face Hub with nothing but PyTorch
and diffusers.

> **Code is Apache-2.0.** Released models are trained on datasets with their own
> terms. In particular the **Waymo models are non-commercial** (Waymo Open Dataset
> License) — see [Licensing](#licensing), `WAYMO_NOTICE.md`, and
> `THIRD_PARTY_LICENSES.md`.

## Roadmap

✅ Inference code (`worldflow3d` package)

✅ Front3D (indoor) checkpoints

✅ Waymo (WOD) checkpoints

⬜ PyPI package (`pip install worldflow3d`) — early July 2026

⬜ Map processing (raw scenes → conditioning maps) — mid July 2026

⬜ Training code — mid July 2026

⬜ Additional indoor layouts — end of July 2026

## Install

```bash
pip install worldflow3d            # core (torch, diffusers, trimesh, scikit-image, ...)
pip install "worldflow3d[layout]"  # + h5py, for loading Front3D/ScanNet++ .h5 layouts
```

PyTorch is expected to be installed for your CUDA/CPU platform. Requires
`torch>=2.2` and `diffusers>=0.29`; tested on **torch 2.2 / diffusers 0.29.2**,
**torch 2.5 / diffusers 0.38**, and **torch 2.12 / diffusers 0.38 (CUDA 13, aarch64)**
(NumPy 1.x and 2.x both work).

<details>
<summary>Running on aarch64 / DGX Spark (GB10)</summary>

On an aarch64 host with a Blackwell GB10 (CUDA 13), the standard PyPI/conda torch
wheels don't target the platform — install torch from the CUDA-13 (`cu130`) index
first, then the package:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "worldflow3d[examples]"   # + tyro, for the CLI example scripts
pip install accelerate                # optional: quiets the model-load warning
```

Only the NVIDIA driver is needed — no separate CUDA toolkit — since the `cu130`
wheel ships its own CUDA runtime. For the fast path on GB10, add `--simultaneous
--compile` to the CLI (Python: `simultaneous=True`, `compile=True` on `from_hub` /
`from_converted`).

Reference timings on GB10 (Waymo coarse→refine, `cfg_scale=1.5`, 30+30 steps,
`smaller_map=True, fraction=0.12`, segment `1172406780360799916`, 1 coarse + 49
refine chunks): `simultaneous=True` + `compile=True` ~21 min,
`simultaneous=False` + `compile=True` ~50 min,  `simultaneous=False` +
`compile=False` ~2h20m

GB10 is memory-bandwidth-bound here (~244 GB/s unified, vs ~768 GB/s on an A6000
and ~3.35 TB/s on an H100), so treat these as platform reference numbers, not a
general benchmark — discrete GPUs are substantially faster.

</details>

## Quickstart

Each cascade stage is a **subfolder** of a Hub repo (`<dataset>-coarse`, plus
`<dataset>-refine` / `-color`). The Waymo and Front3D cascades live in separate
repos (Waymo is non-commercial — see [Licensing](#licensing)). Load with `from_hub`:

```python
import torch
from worldflow3d import WorldFlow3DPipeline
from worldflow3d.pipeline.datatypes import LayoutContext
from worldflow3d.conditioning.waymo import load_waymo_map_json
from worldflow3d.recon import save_mesh

# Waymo coarse -> refine cascade, straight from the Hub (non-commercial).
pipe = WorldFlow3DPipeline.from_hub(
    "pci-lab/worldflow3d-waymo", stage="waymo-coarse", refinement_stages=["waymo-refine"],
    device="cuda",
)

# A shipped sample map -- self-contained: no coord transform or data root needed.
map_json = load_waymo_map_json("1172406780360799916", map_dir="examples/sample_maps")
ctx = LayoutContext("waymo", "1172406780360799916", map_json)

result = pipe(layout_context=ctx, cfg_scale=1.5, sampling_steps=30,
              refine=True, refine_sampling_steps=30, use_uniform_chunking=True,
              simultaneous=True, smaller_map=True, fraction=0.12)

save_mesh(result.voxels.cpu(), result.voxel_size, "scene.ply")
```

CLI (`--repo`/`--stage` for the Hub, or `--model-dir` for local dirs):

```bash
python -m worldflow3d.scripts.generate_streets \
    --repo pci-lab/worldflow3d-waymo --stage waymo-coarse --refine-stage waymo-refine \
    --segment 1172406780360799916 --map-dir examples/sample_maps \
    --cfg-scale 1.5 --coarse-steps 30 --refine-steps 30 \
    --smaller-map --fraction 0.12 --use-uniform-chunking --simultaneous \
    --output outputs/segment-1172406780360799916.ply
```

> The sample maps used above ship in the repository under `examples/sample_maps/`
> (they are not bundled in the PyPI wheel). Point `--map-dir` / `map_dir=` at any
> directory of map JSONs.

For a **colored** scene, use the color refinement stage. The color model is
**tag-conditioned** (`location` / `time_of_day` / `weather`) — pass tags for a
coherent result; without them the color channels are unconditioned and the mesh
shows color/surface artifacts. `save_mesh` also writes a colored `<name>_color.ply`.

```bash
python -m worldflow3d.scripts.generate_streets \
    --repo pci-lab/worldflow3d-waymo --stage waymo-coarse --refine-stage waymo-color \
    --segment 1172406780360799916 --map-dir examples/sample_maps \
    --cfg-scale 1.5 --coarse-steps 30 --refine-steps 30 \
    --smaller-map --fraction 0.12 --use-uniform-chunking --simultaneous \
    --coarse-tag location=location_sf \
    --refine-tag time_of_day=Dawn/Dusk --refine-tag weather=sunny \
    --output outputs/segment-1172406780360799916_color.ply
```

From Python, pass `tags=` / `refine_tags=` to the pipeline, e.g.
`refinement_stages=["waymo-color"]` with
`tags={"location": "location_sf"}, refine_tags={"time_of_day": "Dawn/Dusk", "weather": "sunny"}`.
Valid values: `location` ∈ {location_sf, location_phx, location_other};
`time_of_day` ∈ {Dawn/Dusk, Day, Night}; `weather` ∈ {sunny, rain}.

Indoor (Front3D), layout-conditioned coarse → color refinement:

```bash
python -m worldflow3d.scripts.generate_indoor \
    --repo pci-lab/worldflow3d --stage front3d-coarse --refine-stage front3d-color \
    --scene-id <uuid> --data-root front3d_scenes --refine --output outputs/scene.ply
```

## Models / repo layout

Each stage is a self-contained diffusers layout in its own subfolder. The model
**code** is supplied by this package, so the subfolders carry only weights +
config (no duplicated `.py`). The Waymo and Front3D cascades are in **separate
repos** because they carry different licenses:

```
pci-lab/worldflow3d-waymo/   (NON-COMMERCIAL — Waymo Open Dataset License)
├── waymo-coarse/     model_index.json  unet/{config.json, *.safetensors}  scheduler/
├── waymo-refine/     "    (source-flow refinement, geometry only)
└── waymo-color/      "    (source-flow refinement, geometry + color, unet in=4)

pci-lab/worldflow3d/         (Front3D — 3D-FRONT terms)
├── front3d-coarse/   "
└── front3d-color/    "    (source-flow + color, unet in=4)
```

`waymo-refine` and `waymo-color` are **alternative** refinement stages for the same
coarse output — pick one. `waymo-color` also predicts per-voxel color, so `save_mesh`
writes an extra colored `<name>_color.ply` sidecar. Each `unet/` folder is a distinct
trained model (different weights and configs), not a copy.

## Loading

| Source | Call | Needs |
|---|---|---|
| **Hub (cascade)** | `WorldFlow3DPipeline.from_hub(repo, stage, refinement_stages=[...])` | `pip install worldflow3d` |
| **Local dirs** | `WorldFlow3DPipeline.from_converted(coarse_dir, [refine_dir, ...])` | converted dirs on disk |
| **Bare diffusers** | `DiffusionPipeline.from_pretrained(repo, trust_remote_code=True)` | a model at the **repo root** only |

> **Why no `subfolder=` one-liner?** `diffusers.DiffusionPipeline.from_pretrained`
> does not apply `subfolder` at the *pipeline* level — it reads `model_index.json`
> only from the repo root (true even on the latest diffusers). So the
> subfolder/cascade layout is loaded via `from_hub` (this package). The bare-diffusers
> `trust_remote_code` one-liner works only for a model published at a repo root.

## Design (diffusers-native)

| Component | Class | Notes |
|---|---|---|
| Model | `WorldFlow3DUNet(ModelMixin, ConfigMixin)` | wraps the 3D UNet; `save_pretrained`/`from_pretrained` (safetensors) |
| Scheduler | `WorldFlow3DFlowScheduler(SchedulerMixin, ConfigMixin)` | velocity-prediction Euler flow matching |
| Pipeline | `WorldFlow3DPipeline(DiffusionPipeline)` | custom `__call__`: chunking, CFG, source-flow + standard reconstruction, coarse→refine cascade, sequential + simultaneous |

Default models are **direct-diffusion** (voxel-space; no VAE). A VAE-decode path is
wired but optional. Classifier-free guidance, the model-time scaling, and the
inpainting blend live in the pipeline; the scheduler does only the per-step Euler update.

## Licensing

WorldFlow3D ships under multiple licenses depending on the artifact:

| Artifact | License |
|---|---|
| **Code** (`worldflow3d` package) | Apache-2.0 (`LICENSE`) |
| **Waymo** models (`waymo-coarse`, `waymo-refine`) + sample maps | **Waymo Dataset License — non-commercial** (`WAYMO_NOTICE.md`, <https://waymo.com/open/terms>) |
| **Front3D** models (`front3d-coarse`, `front3d-color`) | 3D-FRONT terms (non-commercial) |
| Scenes / meshes / datasets generated with the **Waymo** models | "Distributed WOD Models" — also non-commercial |

The Apache-2.0 license covers the code only; it does **not** grant rights to the
Waymo- or Front3D-derived weights. Using the Waymo models or sample maps means
agreeing to the Waymo Open Dataset terms (non-commercial), providing the required
attribution, and passing the notice on to downstream recipients. See
`WAYMO_NOTICE.md` and `THIRD_PARTY_LICENSES.md`.

## Supported

- **Waymo** (map-conditioned) coarse→refine cascade — **sequential** and
  **simultaneous** (feather-averaged) generation.
- **Front3D** (layout-conditioned) coarse→refine cascade including **source-flow +
  color** (unet in=4, with a color mesh sidecar).
- Hub loading via `from_hub` (subfolder cascade) on both supported stacks
  (torch 2.2 / diffusers 0.29 and torch 2.5 / diffusers 0.38; NumPy 1.x and 2.x).

## Citation

If you use WorldFlow3D in your research, please cite:

```bibtex
@inproceedings{joshi2026worldflow3d,
  title     = {WorldFlow3D: Flowing Through 3D Distributions for Unbounded World Generation},
  author    = {Joshi, Amogh and Ost, Julian and Heide, Felix},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

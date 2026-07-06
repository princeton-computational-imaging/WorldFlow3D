# Third-Party Licenses & Attributions

WorldFlow3D is released under the **Apache License 2.0** (see `LICENSE`). It
incorporates, vendors, or depends on the third-party components listed below,
each of which remains under its own license. This file is the authoritative
record required by the Apache-2.0 `NOTICE` clause.

> **Compatibility:** all bundled submodules are MIT-licensed, which is
> compatible with redistribution under Apache-2.0. Runtime Python dependencies
> are installed by the user via `pip`/`conda` and are not redistributed here.

## Git submodules (`third_party/`)

| Submodule | Upstream | License | Used by | Needed for |
|-----------|----------|---------|---------|------------|
| `sdf-gen` | https://github.com/amogh7joshi/sdf-gen | MIT | `scripts/data/export_volume_*.py` | **Data preprocessing only** (UDF generation). Not needed for inference or training. |
| `concerto` | https://github.com/Pointcept/Concerto | MIT | `src/models/encoders/foundation.py`, `scripts/inference/fd_concerto.py` | **Optional** feature supervision (VAE training) and the FD-Concerto metric. Imported behind `try/except`. |
| `utonia` | https://github.com/Pointcept/Utonia | see upstream | `scripts/inference/run_utonia.py` | **Optional** feature visualization. Not part of the core pipeline. |

> **TODO before public release:** vendor a copy of each submodule's `LICENSE`
> text under `third_party/<name>/LICENSE` and confirm the Utonia upstream
> license terms. `nksr` (used only by the `[dataprep]` Waymo mesh-reconstruction
> stage) ships under its own license — verify and record it here before
> documenting that path.

## Key runtime dependencies (installed via pip; not redistributed)

| Package | License | Role |
|---------|---------|------|
| PyTorch (`torch`) | BSD-3-Clause | core tensor / autograd runtime |
| `diffusers` | Apache-2.0 | model/scheduler/pipeline mixins; HF Hub loading |
| `huggingface_hub`, `safetensors` | Apache-2.0 | weight hosting + serialization |
| `numpy`, `scipy` | BSD-3-Clause | numerics |
| `scikit-image` | BSD-3-Clause | marching cubes (mesh extraction) |
| `trimesh` | MIT | mesh I/O |
| `tyro` | MIT | CLI parsing |
| `einops` | MIT | tensor reshaping |
| `timm` | Apache-2.0 | optional DINOv3/DINOv2 feature extractors `[features]` |
| `gsplat`, `nerfstudio` | Apache-2.0 | optional 3D Gaussian Splatting `[gaussian]` |
| `nksr`, `pycg`, `meshlib`, `open3d`, `webdataset` | see upstream | optional Waymo mesh reconstruction `[dataprep]` |

## Pretrained models & datasets (separate terms — NOT granted by this license)

- **Waymo Open Dataset** — the released Waymo model checkpoints
  (`waymo-coarse`, `waymo-refine`) are *Distributed WOD Models* and the sample
  maps in `examples/sample_maps/` are De Minimis extracts; both are derived from
  WOD and released for **non-commercial use only** under the Waymo Dataset
  License Agreement (NOT this Apache-2.0 license). Generated scenes/datasets
  inherit those terms. See `WAYMO_NOTICE.md` and <https://waymo.com/open/terms>.
  The sample maps are excluded from the PyPI package.
- **3D-FRONT / 3D-FUTURE** — governed by their own non-commercial terms.
- **ScanNet++** — governed by its own terms of use.
- **DINOv3 / DINOv2** (via `timm` / Hugging Face) — governed by Meta's model
  licenses.

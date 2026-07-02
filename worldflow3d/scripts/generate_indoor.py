"""Minimal indoor (Front3D / ScanNet++) generation CLI for the WorldFlow3D package.

Loads a converted model (coarse + optional refinement) and generates an indoor
scene, then exports a ``.ply`` mesh. Two modes:

* **Layout-conditioned** (``--scene-id``): loads the Front3D layout (+ furniture
  bboxes) for the scene, derives the scene extent from the UDF volume shape, and
  conditions generation on it. Only works if the converted coarse model has
  ``has_layout=True`` in its ``level_meta`` (else layout is ignored and the run is
  effectively unconditional over the scene's footprint).
* **Unconditional** (``--output-size X Y``): generates a fixed-size scene (in
  *finest*-resolution voxels) with grid-tiled coarse generation, no layout.

Examples:
    # Layout-conditioned Front3D scene from the Hub (coarse + color refinement)
    python -m worldflow3d.scripts.generate_indoor \
        --repo pci-lab/worldflow3d --stage front3d-coarse --refine-stage front3d-color \
        --scene-id 00004f89-9aa5-43c2-ae3c-129586be8aaa \
        --data-root /path/to/front3d_scenes \
        --refine --output outputs/front3d/00004f89.ply

    # Unconditional 512x512 (finest voxels) scene from local dirs, coarse only
    python -m worldflow3d.scripts.generate_indoor \
        --model-dir models_diffusers/front3d_coarse \
        --output-size 512 512 --output outputs/front3d/uncond.ply

Note: the coarse model in ``front3d_flow_context_eccv`` is direct-diffusion with
NO layout conditioning, so it runs unconditionally even when ``--scene-id`` is
given (the scene id is then only used to size the scene extent from its UDF).
"""
import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch

from worldflow3d.conditioning import load_layout_data
from worldflow3d.pipeline.datatypes import LayoutContext
from worldflow3d.pipeline.pipeline import WorldFlow3DPipeline
from worldflow3d.recon import save_mesh


def _compute_centered_scene_extent_and_offset(
    scene_shape: Tuple[int, int, int],
    chunk_size: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int]]]:
    """Center a smaller layout inside at least one chunk.

    Indoor ``scene_shape`` is ``(Z_floor, Y_height, X_floor)`` (UDF npy order),
    while ``chunk_size`` is ``(X_floor, Z_height, Y_floor)``. For the first-level
    indoor path the floor dims map to ``scene_extent=(scene_z, chunk_h, scene_x)``.

    Returns ``(scene_extent, region_offset)`` where ``region_offset`` is
    ``(-pad_z, -pad_x)`` (in coarse voxels) when the layout is smaller than one
    chunk, else ``None``. Ported from ``scripts/inference/generate_indoor.py``.
    """
    scene_z, _scene_h, scene_x = scene_shape
    chunk_floor_z = int(chunk_size[0])
    chunk_floor_x = int(chunk_size[2])
    padded_scene_z = max(int(scene_z), chunk_floor_z)
    padded_scene_x = max(int(scene_x), chunk_floor_x)
    scene_extent = (padded_scene_z, int(chunk_size[1]), padded_scene_x)

    pad_before_z = max(0, (padded_scene_z - int(scene_z)) // 2)
    pad_before_x = max(0, (padded_scene_x - int(scene_x)) // 2)
    if pad_before_z == 0 and pad_before_x == 0:
        return scene_extent, None
    return scene_extent, (-pad_before_z, -pad_before_x)


def _build_layout_context(
    dataset_type: str,
    scene_id: str,
    data_root: str,
    coarse_voxel_size: float,
    coarse_chunk: Tuple[int, int, int],
    layout_dir: Optional[str],
    bbox_dir: Optional[str],
) -> Tuple[LayoutContext, Tuple[int, int, int]]:
    """Load layout + bboxes for a scene and build a LayoutContext + scene_extent.

    Mirrors ``scripts/inference/generate_indoor.py``: ``scene_shape`` comes from
    the coarse UDF volume ``.npy`` shape (``(Z, H, X)``); layout/bbox come from
    ``<data_root>/layout/<id>.h5`` and ``<data_root>/bbox/<id>.h5`` (overridable).
    """
    layout_root = layout_dir or os.path.join(data_root, "layout")
    bbox_root = bbox_dir or os.path.join(data_root, "bbox")
    layout_path = os.path.join(layout_root, f"{scene_id}.h5")
    bbox_path = os.path.join(bbox_root, f"{scene_id}.h5")
    raw_data = load_layout_data(dataset_type, layout_path=layout_path, bbox_path=bbox_path)

    voxel_str = f"udf_voxel_{coarse_voxel_size}"
    udf_path = os.path.join(data_root, voxel_str, f"{scene_id}.npy")
    scene_shape = tuple(int(s) for s in np.load(udf_path, mmap_mode="r").shape)

    scene_extent, region_offset = _compute_centered_scene_extent_and_offset(
        scene_shape, coarse_chunk
    )
    ctx = LayoutContext(
        dataset_type=dataset_type,
        source_id=scene_id,
        raw_data=raw_data,
        scene_shape=scene_shape,
        region_offset=region_offset,
    )
    return ctx, scene_extent


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Hub source (one repo, cascade as subfolders) ...
    ap.add_argument("--repo", default=None, help="HF Hub repo id (e.g. pci-lab/worldflow3d)")
    ap.add_argument("--stage", default="front3d-coarse", help="coarse-stage subfolder in --repo")
    ap.add_argument("--refine-stage", action="append", default=[],
                    help="refinement-stage subfolder in --repo (repeatable)")
    # ... OR local converted dirs
    ap.add_argument("--model-dir", default=None, help="converted coarse model dir (if not using --repo)")
    ap.add_argument("--refinement-dir", action="append", default=[],
                    help="converted refinement model dir (repeatable)")
    ap.add_argument("--scene-id", default=None,
                    help="Front3D scene UUID for layout conditioning + extent")
    ap.add_argument("--dataset-type", default="front3d", choices=["front3d", "scannetpp"])
    ap.add_argument("--data-root", default=None,
                    help="dir with layout/, bbox/, udf_voxel_<vs>/ (required with --scene-id)")
    ap.add_argument("--layout-dir", default=None, help="override layout .h5 dir")
    ap.add_argument("--bbox-dir", default=None, help="override bbox .h5 dir")
    ap.add_argument("--output-size", type=int, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="unconditional scene size in FINEST voxels (no layout)")
    ap.add_argument("--finest-voxel-size", type=float, default=None,
                    help="finest voxel size (for --output-size scaling); "
                         "defaults to the refinement model's voxel size, else coarse")
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--coarse-steps", type=int, default=25)
    ap.add_argument("--refine-steps", type=int, default=25)
    ap.add_argument("--coarse-overlap", type=float, default=0.5)
    ap.add_argument("--fine-overlap", type=float, default=0.5)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--use-uniform-chunking", action="store_true")
    ap.add_argument("--one-chunk", action="store_true")
    ap.add_argument("--simultaneous", action="store_true",
                    help="generate all chunks at once with feather-averaged velocities (flow only)")
    ap.add_argument("--simultaneous-batch-size", type=int, default=4)
    ap.add_argument("--feather-margin", type=int, default=8)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile() each stage's UNet (one-time warmup, faster steps after)")
    ap.add_argument("--store-coarse", action="store_true",
                    help="also export the coarse mesh when refining")
    ap.add_argument("--crop-top", type=float, default=0.1,
                    help="fraction of the top of the volume to crop in the mesh")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not args.scene_id and not args.output_size:
        ap.error("provide either --scene-id (layout) or --output-size (unconditional)")
    if args.scene_id and not (args.data_root or args.layout_dir):
        ap.error("--scene-id needs --data-root (or --layout-dir/--bbox-dir)")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.repo:
        pipe = WorldFlow3DPipeline.from_hub(
            args.repo, stage=args.stage, refinement_stages=args.refine_stage or None, device=device,
            compile=args.compile,
        )
        has_refine = bool(args.refine_stage)
    else:
        if not args.model_dir:
            ap.error("provide either --repo (Hub) or --model-dir (local converted dir)")
        pipe = WorldFlow3DPipeline.from_converted(
            args.model_dir, refinement_dirs=args.refinement_dir or None, device=device,
            compile=args.compile,
        )
        has_refine = bool(args.refinement_dir)

    coarse_meta = pipe._meta(pipe.unet)
    coarse_voxel_size = float(coarse_meta["voxel_size"])
    coarse_chunk = tuple(int(v) for v in coarse_meta["output_chunk_size"])  # (X, Z, Y)

    # ---- scene extent + (optional) layout context ----
    layout_ctx = None
    if args.scene_id is not None:
        layout_ctx, scene_extent = _build_layout_context(
            args.dataset_type, args.scene_id, args.data_root, coarse_voxel_size,
            coarse_chunk, args.layout_dir, args.bbox_dir,
        )
        if not coarse_meta.get("has_layout"):
            print("[warn] coarse model has no layout conditioning; --scene-id "
                  "only sizes the scene extent (generation is unconditional).")
    else:
        # Unconditional: --output-size is in finest voxels -> scale to coarse.
        finest_vs = args.finest_voxel_size
        if finest_vs is None:
            if pipe._refinements:
                finest_vs = float(pipe._meta(pipe._refinements[-1][0])["voxel_size"])
            else:
                finest_vs = coarse_voxel_size
        scene_x = int(args.output_size[0] * finest_vs / coarse_voxel_size)
        scene_y = int(args.output_size[1] * finest_vs / coarse_voxel_size)
        scene_x = ((scene_x + coarse_chunk[0] - 1) // coarse_chunk[0]) * coarse_chunk[0]
        scene_y = ((scene_y + coarse_chunk[2] - 1) // coarse_chunk[2]) * coarse_chunk[2]
        scene_extent = (scene_x, coarse_chunk[1], scene_y)

    print(f"[indoor] scene_extent={scene_extent} coarse_vs={coarse_voxel_size} "
          f"chunk={coarse_chunk} layout={layout_ctx is not None}")

    out = pipe(
        layout_context=layout_ctx,
        scene_extent=scene_extent,
        cfg_scale=args.cfg_scale,
        sampling_steps=args.coarse_steps,
        refine_sampling_steps=args.refine_steps,
        overlap_ratio=args.coarse_overlap,
        fine_overlap_ratio=args.fine_overlap,
        use_uniform_chunking=args.use_uniform_chunking,
        one_chunk=args.one_chunk,
        simultaneous=args.simultaneous,
        simultaneous_batch_size=args.simultaneous_batch_size,
        feather_margin=args.feather_margin,
        refine=bool(args.refine and has_refine),
        return_coarse=bool(args.store_coarse and args.refine and has_refine),
    )

    if isinstance(out, tuple):
        result, coarse = out
    else:
        result, coarse = out, None

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_mesh(result.voxels.cpu(), result.voxel_size, args.output, crop_top=args.crop_top)
    print(f"saved {args.output} (voxel_size={result.voxel_size}, "
          f"chunks={len(result.surface_mapping)})")

    if coarse is not None:
        base, ext = os.path.splitext(args.output)
        coarse_path = f"{base}_coarse{ext}"
        save_mesh(coarse.voxels.cpu(), coarse.voxel_size, coarse_path, crop_top=args.crop_top)
        print(f"saved {coarse_path} (voxel_size={coarse.voxel_size})")


if __name__ == "__main__":
    main()

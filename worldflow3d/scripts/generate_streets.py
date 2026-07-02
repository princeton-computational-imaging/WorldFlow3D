"""Minimal street-scene generation CLI for the WorldFlow3D diffusers package.

Generates outdoor, map-conditioned street scenes: loads a model (coarse + optional
refinement) from the Hugging Face Hub (one repo, cascade stages as subfolders) or
from local converted dirs, conditions on a vector street map, generates a scene,
and exports a .ply mesh. The released models and the shipped sample maps use the
Waymo Open Dataset map format (the method itself is not dataset-specific).

Examples:
    # from the Hub, using a shipped sample map (self-contained -- no data root needed)
    python -m worldflow3d.scripts.generate_streets \
        --repo pci-lab/worldflow3d-waymo --stage waymo-coarse --refine-stage waymo-refine \
        --segment 1020629352036937500 --map-dir examples/sample_maps \
        --cfg-scale 1.5 --coarse-steps 30 --refine-steps 30 \
        --smaller-map --fraction 0.12 --use-uniform-chunking --simultaneous --compile \
        --output outputs/scene.ply

    # from local converted dirs, against a full map directory (short id + metadata)
    python -m worldflow3d.scripts.generate_streets \
        --model-dir models_diffusers/coarse --refinement-dir models_diffusers/refine \
        --segment 102062 --map-dir /path/to/maps --data-root /path/to/scenes \
        --simultaneous --compile --output outputs/scene.ply

Needs only the vector map JSON. With a shipped sample map, pass its map id as --segment
and omit --data-root; for a full map directory, pass a short segment id plus --data-root
(which holds metadata/segment_names/ to resolve it). The released street models are trained
on the Waymo Open Dataset and are non-commercial -- see the repo LICENSE notes, WAYMO_NOTICE.md,
and the Waymo Open Dataset terms.
"""
import argparse
import os
import random

import numpy as np
import torch

from worldflow3d.conditioning.waymo import load_waymo_map_json
from worldflow3d.pipeline.datatypes import LayoutContext
from worldflow3d.pipeline.pipeline import WorldFlow3DPipeline
from worldflow3d.recon import save_mesh


def _parse_tags(items):
    if not items:
        return None
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Hub source (one repo, cascade as subfolders) ...
    ap.add_argument("--repo", default=None, help="HF Hub repo id (e.g. pci-lab/worldflow3d)")
    ap.add_argument("--stage", default="waymo-coarse", help="coarse-stage subfolder in --repo")
    ap.add_argument("--refine-stage", action="append", default=[],
                    help="refinement-stage subfolder in --repo (repeatable)")
    # ... OR local converted dirs
    ap.add_argument("--model-dir", default=None, help="converted coarse model dir (if not using --repo)")
    ap.add_argument("--refinement-dir", action="append", default=[],
                    help="converted refinement model dir (repeatable)")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--map-dir", required=True, help="dir with WOD_*.json map files")
    ap.add_argument("--data-root", default=None,
                    help="optional: data root with metadata/segment_names/ to resolve short "
                         "segment ids; omit to match sample maps directly by their WOD id")
    ap.add_argument("--coarse-voxel-size", type=float, default=0.4)
    ap.add_argument("--cfg-scale", type=float, default=1.5)
    ap.add_argument("--coarse-steps", type=int, default=30)
    ap.add_argument("--refine-steps", type=int, default=30)
    ap.add_argument("--coarse-overlap", type=float, default=0.3)
    ap.add_argument("--fine-overlap", type=float, default=0.25)
    ap.add_argument("--smaller-map", action="store_true")
    ap.add_argument("--fraction", type=float, default=0.12)
    ap.add_argument("--use-uniform-chunking", action="store_true")
    ap.add_argument("--one-chunk", action="store_true")
    ap.add_argument("--simultaneous", action="store_true",
                    help="generate all chunks at once with feather-averaged velocities (flow only)")
    ap.add_argument("--simultaneous-batch-size", type=int, default=4)
    ap.add_argument("--feather-margin", type=int, default=8)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile() each stage's UNet (one-time warmup, faster steps after)")
    ap.add_argument("--coarse-tag", action="append", default=None, help="k=v (repeatable)")
    ap.add_argument("--refine-tag", action="append", default=None, help="k=v (repeatable)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)  # makes random-default tag selection reproducible

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.repo:
        pipe = WorldFlow3DPipeline.from_hub(
            args.repo, stage=args.stage, refinement_stages=args.refine_stage or None, device=device,
            compile=args.compile,
        )
        do_refine = bool(args.refine_stage)
    else:
        if not args.model_dir:
            ap.error("provide either --repo (Hub) or --model-dir (local converted dir)")
        pipe = WorldFlow3DPipeline.from_converted(
            args.model_dir, refinement_dirs=args.refinement_dir or None, device=device,
            compile=args.compile,
        )
        do_refine = bool(args.refinement_dir)

    map_json = load_waymo_map_json(args.segment, args.map_dir, data_root=args.data_root)
    ctx = LayoutContext("waymo", args.segment, map_json)

    result = pipe(
        layout_context=ctx,
        tags=_parse_tags(args.coarse_tag),
        refine_tags=_parse_tags(args.refine_tag),
        cfg_scale=args.cfg_scale,
        sampling_steps=args.coarse_steps,
        refine_sampling_steps=args.refine_steps,
        overlap_ratio=args.coarse_overlap,
        fine_overlap_ratio=args.fine_overlap,
        smaller_map=args.smaller_map,
        fraction=args.fraction,
        use_uniform_chunking=args.use_uniform_chunking,
        one_chunk=args.one_chunk,
        simultaneous=args.simultaneous,
        simultaneous_batch_size=args.simultaneous_batch_size,
        feather_margin=args.feather_margin,
        refine=do_refine,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_mesh(result.voxels.cpu(), result.voxel_size, args.output)
    print(f"saved {args.output} (voxel_size={result.voxel_size}, "
          f"chunks={len(result.surface_mapping)})")


if __name__ == "__main__":
    main()

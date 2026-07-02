"""WorldFlow3DPipeline — diffusers-native hierarchical 3D scene generation.

Faithful port of ``src/inference/scene_pipeline.py:ScenePipeline.generate_level``
(+ the flow sampler loop) onto a ``DiffusionPipeline`` with a custom ``__call__``.
Classifier-free guidance, the ``1000*t`` model-time scaling, and the inpainting
blend live here (the scheduler only does the per-step Euler update).

All per-level metadata is read from ``unet.config`` (``level_meta`` + ``level_spec``)
written by the converter, so the package needs no research config system.

Both chunk-generation modes are implemented: sequential (the default; chunks are
generated one at a time with inpainting overlap, and the seed-matched parity path)
and simultaneous (``simultaneous=True``; every chunk is integrated through the flow
ODE at once, with per-step velocities feather-averaged across chunks — see
``.simultaneous``).
"""
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

from ..conditioning import get_local_layout_features, voxelize_layout
from ..conditioning.tags import prepare_tag_conditioning, resolve_tag_dict
from ..conditioning.waymo import (
    generate_chunks_from_polyline_samples_with_graph,
    get_local_waymo_map_features,
    get_map_chunk,
    recompute_transform_for_map,
    sample_points_along_polylines,
    vector_to_voxel_map_waymo,
)
from ..recon import (
    find_lowest_map_chunk_surface,
    get_surface_threshold,
    sample_finer_surface_mapping_positions,
    transform_surface_mapping_positions,
    truncate_udf,
)
from .datatypes import LayoutContext, LevelResult
from .scheduler import WorldFlow3DFlowScheduler
from .simultaneous import prepare_chunk_infos, run_simultaneous
from .unet import WorldFlow3DUNet

logger = logging.getLogger("worldflow3d.pipeline")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x


# ---------------------------------------------------------------------------
# Chunk-position helpers (ported verbatim from scene_pipeline.py)
# ---------------------------------------------------------------------------

def _filter_to_center(positions, fraction=0.02):
    coords = np.array(positions)
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2
    extent = coords.max(axis=0) - coords.min(axis=0)
    max_dist = np.linalg.norm(extent) * np.sqrt(fraction) / 2
    return [(x, y) for x, y in positions
            if np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= max_dist]


def _calculate_grid_positions_spiral(scene_extent, chunk_extent, overlap_ratio=0.5):
    step = tuple(max(1, int(c * (1 - overlap_ratio))) for c in chunk_extent)
    seen, positions = set(), []
    x = 0
    while x < scene_extent[0]:
        x_clamped = min(x, scene_extent[0] - chunk_extent[0])
        y = 0
        while y < scene_extent[1]:
            y_clamped = min(y, scene_extent[1] - chunk_extent[1])
            if (x_clamped, y_clamped) not in seen:
                seen.add((x_clamped, y_clamped))
                positions.append((x_clamped, y_clamped))
            if y >= scene_extent[1] - chunk_extent[1]:
                break
            y += step[1]
        if x >= scene_extent[0] - chunk_extent[0]:
            break
        x += step[0]
    cx, cy = scene_extent[0] // 2, scene_extent[1] // 2
    positions.sort(key=lambda p: (p[0] + chunk_extent[0] // 2 - cx) ** 2
                   + (p[1] + chunk_extent[1] // 2 - cy) ** 2)
    return positions


def _compute_occupied_xy_bounds(coarse_voxels, coarse_voxel_size, fine_voxel_size, threshold):
    geom = coarse_voxels[0, 0]
    occupied = (geom < threshold).any(dim=1)
    xs, ys = torch.where(occupied)
    if len(xs) == 0:
        return np.zeros((0, 2))
    ratio = coarse_voxel_size / fine_voxel_size
    return np.stack([(xs.cpu().numpy() * ratio).astype(int),
                     (ys.cpu().numpy() * ratio).astype(int)], axis=1)


def _ensure_coverage(chunk_boxes, occupied_fine_coords, chunk_size, overlap_ratio=0.3):
    cx, cy = chunk_size[0], chunk_size[2]
    if len(occupied_fine_coords) == 0:
        return chunk_boxes
    kept = []
    for box in chunk_boxes:
        x0, y0, x1, y1 = box
        inside = ((occupied_fine_coords[:, 0] >= x0) & (occupied_fine_coords[:, 0] < x1) &
                  (occupied_fine_coords[:, 1] >= y0) & (occupied_fine_coords[:, 1] < y1))
        if inside.any():
            kept.append(box)
    covered = np.zeros(len(occupied_fine_coords), dtype=bool)
    for box in kept:
        x0, y0, x1, y1 = box
        covered |= ((occupied_fine_coords[:, 0] >= x0) & (occupied_fine_coords[:, 0] < x1) &
                    (occupied_fine_coords[:, 1] >= y0) & (occupied_fine_coords[:, 1] < y1))
    stride_x = max(1, int(cx * (1 - overlap_ratio)))
    stride_y = max(1, int(cy * (1 - overlap_ratio)))
    for idx in np.where(~covered)[0]:
        px, py = occupied_fine_coords[idx]
        gx = (int(px) // stride_x) * stride_x
        gy = (int(py) // stride_y) * stride_y
        box = (gx, gy, gx + cx, gy + cy)
        if box not in kept:
            kept.append(box)
    return kept


# ---------------------------------------------------------------------------
# Coarse-geometry conditioning + reconstruction (ported)
# ---------------------------------------------------------------------------

def _setup_coarse_conditioning(coarse_voxels, meta, x_coord, y_coord, z_offset,
                               chunk_extent, chunk_z, factor, full_noise, xi, yi,
                               device, truncation, fine_voxel_size, coarse_voxel_size):
    """Extract coarse-geometry conditioning / source-flow initial sample for a chunk."""
    direct_diffusion = meta["direct_diffusion"]
    use_source_flow = meta["initial_distribution"] != "noise"
    H_channels = meta["H_channels"]
    source_noise_std = meta.get("source_noise_std", 0.0) or 0.0
    use_color = meta.get("use_color", False)

    coord_ratio = fine_voxel_size / coarse_voxel_size
    cx, cy = chunk_extent

    coarse_x = int(round(x_coord * coord_ratio))
    coarse_y = int(round(y_coord * coord_ratio))
    coarse_z = int(round(z_offset * coord_ratio))
    crop_x = int(round(cx * factor * coord_ratio))
    crop_z = int(round(chunk_z * factor * coord_ratio))
    crop_y = int(round(cy * factor * coord_ratio))

    max_x, max_z, max_y = coarse_voxels.shape[2], coarse_voxels.shape[3], coarse_voxels.shape[4]
    x_end = min(coarse_x + crop_x, max_x)
    z_end = min(coarse_z + crop_z, max_z)
    y_end = min(coarse_y + crop_y, max_y)
    L_crop = coarse_voxels[:, :, coarse_x:x_end, coarse_z:z_end, coarse_y:y_end]

    expected = (crop_x, crop_z, crop_y)
    if L_crop.shape[2:] != expected:
        pad_x = expected[0] - L_crop.shape[2]
        pad_z = expected[1] - L_crop.shape[3]
        pad_y = expected[2] - L_crop.shape[4]
        L_crop = F.pad(L_crop, (0, pad_y, 0, pad_z, 0, pad_x), value=truncation)

    geom = L_crop[:, 0:1]
    latent_shape = (1, H_channels, cx, chunk_z, cy)

    if use_source_flow and direct_diffusion:
        coarse_geom = truncate_udf(geom, truncation=truncation)
        normalized_geom = (coarse_geom - 0.5) * 2.0
        if coarse_voxels.shape[1] == 1:
            geom_up = F.interpolate(normalized_geom, size=latent_shape[2:], mode="trilinear", align_corners=False)
            if source_noise_std > 0:
                geom_up = geom_up + source_noise_std * torch.randn_like(geom_up)
            if H_channels > 1:
                color_noise = full_noise[:, 1:, xi:xi + cx, z_offset // factor:z_offset // factor + chunk_z, yi:yi + cy]
                if color_noise.shape[2:] != (cx, chunk_z, cy):
                    color_noise = torch.randn(1, H_channels - 1, cx, chunk_z, cy, device=device)
                initial_sample = torch.cat([geom_up.to(device), color_noise], dim=1)
            else:
                initial_sample = geom_up
            return {"coarse_geometry": None, "initial_sample": initial_sample}
        else:
            color = L_crop[:, 1:]
            sample = torch.cat([normalized_geom, color], dim=1).to(device)
            initial_sample = F.interpolate(sample, size=latent_shape[2:], mode="trilinear", align_corners=False)
            return {"coarse_geometry": None, "initial_sample": initial_sample}

    geom_processed = truncate_udf(geom, truncation=truncation).to(device)
    if use_color and L_crop.shape[1] >= 4:
        coarse_geometry = torch.cat([geom_processed, L_crop[:, 1:4].to(device)], dim=1)
    else:
        coarse_geometry = geom_processed
    return {"coarse_geometry": coarse_geometry.contiguous(), "initial_sample": None}


def _reconstruct_voxels(full_latent, meta, direct_diffusion, truncation,
                        coarse_voxels=None, fine_voxel_size=1.0, coarse_voxel_size=1.0):
    """Decode the assembled latent to voxels (direct-diffusion paths)."""
    use_source_flow = meta["initial_distribution"] != "noise"
    if not direct_diffusion:
        raise NotImplementedError(
            "VAE-decode reconstruction is not shipped in this package version "
            "(default models are direct-diffusion). Register a `vae` component to enable."
        )
    if use_source_flow or coarse_voxels is None:
        voxels = (full_latent / 2.0) + 0.5
        voxels[:, 0:1] = truncate_udf(voxels[:, 0:1], truncation=truncation)
        return voxels
    # standard residual denoising
    scale = int(round(coarse_voxel_size / fine_voxel_size))
    coarse_geom = truncate_udf(coarse_voxels[:, 0:1].float().to(full_latent.device), truncation=truncation)
    coarse_up = F.interpolate(coarse_geom, scale_factor=scale, mode="trilinear", align_corners=False)
    tgt = full_latent.shape[2:]
    pad_x = max(0, tgt[0] - coarse_up.shape[2])
    pad_z = max(0, tgt[1] - coarse_up.shape[3])
    pad_y = max(0, tgt[2] - coarse_up.shape[4])
    if pad_x or pad_z or pad_y:
        coarse_up = F.pad(coarse_up, (0, pad_y, 0, pad_z, 0, pad_x), value=truncation)
    coarse_up = coarse_up[:, :, :tgt[0], :tgt[1], :tgt[2]]
    residual = full_latent[:, 0:1].float() * truncation
    fine_geom = truncate_udf(coarse_up + residual, truncation=truncation)
    if full_latent.shape[1] >= 4:
        color = (full_latent[:, 1:4].float() / 2.0) + 0.5
        return torch.cat([fine_geom, color], dim=1)
    return fine_geom


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class WorldFlow3DPipeline(DiffusionPipeline):
    """Hierarchical, chunked, map/tag-conditioned flow-matching scene generator."""

    def __init__(self, unet: WorldFlow3DUNet, scheduler: WorldFlow3DFlowScheduler, vae=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)
        self._refinements: List[Tuple] = []  # [(unet, scheduler, vae), ...]
        if torch.cuda.is_available():
            # Chunk batches are fixed/near-fixed shape -> let cuDNN pick the fastest
            # conv algorithm per shape instead of the default generic heuristic.
            torch.backends.cudnn.benchmark = True

    @classmethod
    def from_converted(cls, model_dir: str, refinement_dirs: Optional[List[str]] = None,
                       device: str = "cuda", compile: bool = False):
        """Build a pipeline from converter output dirs (coarse + optional refinements).

        Each dir is a diffusers layout with ``unet/`` and ``scheduler/`` (and an
        optional ``vae/``). This avoids needing ``trust_remote_code`` / a published
        Hub repo, and is the loader the CLI scripts use locally.

        Args:
            model_dir: converted coarse-stage dir.
            refinement_dirs: ordered converted refinement-stage dirs to chain.
            device: target device.
            compile: wrap each stage's UNet with ``torch.compile`` (same weights;
                fused kernels). Pays a one-time warmup on the first forward per stage
                per distinct input shape; faster steps thereafter.
        """
        import os

        def _load(d):
            unet = WorldFlow3DUNet.from_pretrained(os.path.join(d, "unet"))
            sched = WorldFlow3DFlowScheduler.from_pretrained(os.path.join(d, "scheduler"))
            return unet, sched

        unet, sched = _load(model_dir)
        if compile:
            unet = torch.compile(unet)
        pipe = cls(unet, sched).to(device)
        for rd in (refinement_dirs or []):
            ru, rs = _load(rd)
            if compile:
                ru = torch.compile(ru)
            pipe.add_refinement_level(ru, rs)
        return pipe

    @classmethod
    def from_hub(cls, repo_id: str, stage: Optional[str] = None,
                 refinement_stages: Optional[List[str]] = None, device: str = "cuda",
                 token: Optional[str] = None, cache_dir: Optional[str] = None,
                 compile: bool = False):
        """Load a (cascade) pipeline from ONE Hub repo using the subfolder topology.

        A single repo holds the whole cascade as subfolders, e.g.
        ``waymo-coarse/``, ``waymo-refine/`` (or ``front3d-coarse/``,
        ``front3d-color/``), each a self-contained converted layout.

        diffusers' ``DiffusionPipeline.from_pretrained`` does **not** apply
        ``subfolder`` at the pipeline level (it reads ``model_index.json`` only from
        the repo root), so we ``snapshot_download`` the needed stage subfolders and
        build via :meth:`from_converted` on the local dirs. This is the supported way
        to load the subfolder layout — the bare-``diffusers`` ``trust_remote_code``
        one-liner only works for a model stored at a repo root.

        Args:
            repo_id: ``<user-or-org>/<name>`` Hub repo holding the cascade.
            stage: subfolder of the coarse stage (e.g. ``"waymo-coarse"``). If
                ``None``, the repo root itself is treated as the (single-stage) model.
            refinement_stages: ordered subfolders of refinement stages to chain.
            device, token, cache_dir: passed through.
            compile: ``torch.compile`` each stage's UNet (see :meth:`from_converted`).
        """
        import os

        from huggingface_hub import snapshot_download

        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        stages = [s for s in ([stage] + list(refinement_stages or [])) if s]
        allow = [f"{s}/*" for s in stages] if stages else None
        local = snapshot_download(repo_id, allow_patterns=allow, token=token, cache_dir=cache_dir)
        _dir = lambda s: os.path.join(local, s) if s else local
        return cls.from_converted(
            _dir(stage), [_dir(s) for s in (refinement_stages or [])], device=device,
            compile=compile,
        )

    # -- cascade ----------------------------------------------------------
    def add_refinement_level(self, unet: WorldFlow3DUNet,
                             scheduler: Optional[WorldFlow3DFlowScheduler] = None, vae=None):
        """Register a refinement stage (run after coarse, in insertion order)."""
        scheduler = scheduler or WorldFlow3DFlowScheduler()
        self._refinements.append((unet.to(self.device), scheduler, vae))

    @staticmethod
    def _meta(unet):
        return dict(unet.config.level_meta or {})

    @staticmethod
    def _spec(unet):
        return dict(unet.config.level_spec or {})

    # -- per-chunk flow sampling (CFG + inpainting blend live here) -------
    def _predict_velocity(self, unet, x, t, cond, cfg_scale, coarse_geometry):
        t_tensor = torch.full((x.shape[0],), 1000.0 * float(t), device=x.device, dtype=torch.float32)
        if cfg_scale != 1.0 and cond is not None and "layout" in cond:
            B = x.shape[0]
            x2 = torch.cat([x, x], 0)
            t2 = torch.cat([t_tensor, t_tensor], 0)
            cond2 = {k: (torch.cat([v, torch.zeros_like(v)], 0) if k == "layout"
                         else torch.cat([v, v], 0)) for k, v in cond.items()}
            cg2 = torch.cat([coarse_geometry, coarse_geometry], 0) if coarse_geometry is not None else None
            out = unet(x2, t2, cond=cond2, coarse_geometry=cg2).sample
            v_cond, v_uncond = out[:B], out[B:]
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        return unet(x, t_tensor, cond=cond, coarse_geometry=coarse_geometry).sample

    def _sample_chunk(self, unet, scheduler, shape, cond, cfg_scale, sampling_steps,
                      coarse_geometry=None, initial_sample=None, mask=None, x0=None,
                      noise=None, temperature=1.0):
        device = self.device
        scheduler.set_timesteps(sampling_steps, device=device)
        if initial_sample is not None:
            sample = initial_sample.to(device)
        elif noise is not None:
            sample = noise.to(device)
        else:
            sample = torch.randn(shape, device=device) * temperature

        eps_known = None
        if mask is not None and x0 is not None:
            eps_known = noise.to(device) if noise is not None else torch.randn_like(x0)

        for t in scheduler.timesteps:
            tf = float(t)
            if mask is not None and x0 is not None:
                if initial_sample is not None:
                    x_known = (1 - tf) * x0 + tf * initial_sample.to(x0.device)
                else:
                    x_known = (1 - tf) * x0 + scheduler.sigma(tf) * eps_known
                sample = mask * x_known + (1.0 - mask) * sample
            v = self._predict_velocity(unet, sample, tf, cond, cfg_scale, coarse_geometry)
            sample = scheduler.step(v, tf, sample).prev_sample
        return sample

    # -- one generation level (faithful port of generate_level) ----------
    @torch.no_grad()
    def _generate_level(
        self, unet, scheduler, layout_context=None, scene_extent=None, tags=None,
        cfg_scale=3.0, sampling_steps=50, overlap_ratio=0.3, smaller_map=False,
        fraction=0.1, coarse_voxels=None, prev_surface_mapping=None,
        prev_voxel_size=None, prev_chunk_size=None, prev_factor=1, prev_map_offset=None,
        use_map_chunking=False, use_uniform_chunking=False, one_chunk=False,
        simultaneous=False, simultaneous_batch_size=4, feather_margin=8,
    ) -> LevelResult:
        meta = self._meta(unet)
        device = self.device
        ocs = tuple(int(v) for v in meta["output_chunk_size"])  # (X, Z, Y)
        voxel_size = float(meta["voxel_size"])
        direct_diffusion = bool(meta["direct_diffusion"])
        factor = 1 if direct_diffusion else int(meta["factor"])
        is_first_level = coarse_voxels is None
        truncation = float(meta["truncation"])
        H_channels = int(meta["H_channels"])
        use_source_flow = meta["initial_distribution"] != "noise"
        dataset_type = meta.get("dataset_type")
        has_layout = bool(meta.get("has_layout"))
        layout_cfg = SimpleNamespace(**meta["layout_config"]) if meta.get("layout_config") else None
        has_tags = bool(meta.get("has_tags"))
        tag_cfg = SimpleNamespace(types=meta.get("tag_types") or [])

        resolved_tags = None
        if has_tags and tag_cfg.types:
            # Tag-conditioned level: always resolve so the model gets valid tag
            # channels. User-given tags are used; any unspecified type is filled
            # with a random default (avoids the zero-tag / off-distribution result).
            resolved_tags = resolve_tag_dict(tags, tag_cfg, use_random_defaults=True)
            if tags:
                logger.info("[%s] tags=%s", meta.get("level_name"), resolved_tags)
            else:
                logger.info("[%s] no tags given -> random defaults: %s",
                            meta.get("level_name"), resolved_tags)

        # ---- Step 1: chunk positions ----
        has_map = layout_context is not None and layout_context.dataset_type == "waymo"
        map_transform = map_extent = None
        if has_map:
            if layout_context.transform_data is not None:
                rot = layout_context.transform_data["rotation_matrix"].copy()
            else:
                mat_val = np.sin(np.deg2rad(45))
                rot = np.array([[mat_val, -mat_val, 0., 0.], [mat_val, mat_val, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
            map_transform, map_extent = recompute_transform_for_map(layout_context.raw_data, rot, voxel_size)
        elif is_first_level and scene_extent is None:
            if layout_context is not None and layout_context.scene_shape is not None:
                scene_z, _h, scene_x = layout_context.scene_shape
                scene_extent = (scene_z, ocs[1], scene_x)
            else:
                raise ValueError("first level needs a street-map layout_context, scene_extent, or scene_shape")

        inherited_surface_heights: Dict[Tuple[int, int], int] = {}
        chunk_boxes = None

        if is_first_level and not has_map:
            positions = _calculate_grid_positions_spiral((scene_extent[0], scene_extent[2]),
                                                         (ocs[0], ocs[2]), overlap_ratio)
            map_offset = (0.0, 0.0)
        elif is_first_level or use_map_chunking:
            pts = sample_points_along_polylines(layout_context.raw_data, map_transform, spacing=20.0)
            chunk_boxes = generate_chunks_from_polyline_samples_with_graph(
                pts, ocs, map_extent, overlap_threshold=overlap_ratio * 1.07,
                min_overlap=overlap_ratio, padding=0.2)
            if not is_first_level and coarse_voxels is not None:
                thr = get_surface_threshold(prev_voxel_size)
                occ = _compute_occupied_xy_bounds(coarse_voxels, prev_voxel_size, voxel_size, thr)
                chunk_boxes = _ensure_coverage(chunk_boxes, occ, ocs, overlap_ratio)
            positions = [(b[0], b[1]) for b in chunk_boxes]
            map_offset = (0.0, 0.0)
            if prev_map_offset is not None and prev_voxel_size is not None:
                vr = prev_voxel_size / voxel_size
                map_offset = (prev_map_offset[0] * vr, prev_map_offset[1] * vr)
            if positions and prev_map_offset is None:
                pa = np.array(positions)
                mnx, mny = float(pa[:, 0].min()), float(pa[:, 1].min())
                map_offset = (mnx, mny)
                positions = [(x - mnx, y - mny) for x, y in positions]
            if smaller_map and prev_map_offset is None:
                positions = _filter_to_center(positions, fraction=fraction)
        else:
            prev_sm_output = {(x * prev_factor, y * prev_factor): z * prev_factor
                              for (x, y), z in prev_surface_mapping.items()}
            fn = sample_finer_surface_mapping_positions if use_uniform_chunking else transform_surface_mapping_positions
            inherited = fn(prev_sm_output, prev_voxel_size, tuple(prev_chunk_size),
                           voxel_size, ocs, overlap_ratio)
            positions = [(float(x), float(y)) for x, y in inherited.keys()]
            inherited_surface_heights = {(int(x), int(y)): z for (x, y), z in inherited.items()}
            cx_out, cy_out = ocs[0], ocs[2]
            chunk_boxes = [(x, y, x + cx_out, y + cy_out) for x, y in positions]
            if prev_map_offset is not None and prev_voxel_size is not None:
                vr = prev_voxel_size / voxel_size
                map_offset = (prev_map_offset[0] * vr, prev_map_offset[1] * vr)
            else:
                map_offset = (0.0, 0.0)

        if not positions:
            raise ValueError("no chunk positions generated")
        if one_chunk:
            positions = [positions[0]]

        # ---- Step 2: target extent + full layout voxelization ----
        pa = np.array(positions)
        max_x = int(pa[:, 0].max() + ocs[0])
        max_y = int(pa[:, 1].max() + ocs[2])
        target_extent = [max_x, max_y]
        target_latent = [max_x // factor, max_y // factor]
        chunk_extent = (ocs[0] // factor, ocs[2] // factor)
        chunk_z = ocs[1] // factor

        has_layout_cond = has_layout and layout_context is not None
        voxelized_layout = None
        layout_features_full = None
        if has_layout_cond and dataset_type == "waymo":
            mqh = ocs[1] * 4
            bmin = np.array([map_offset[0], 0, map_offset[1]])
            bmax = np.array([map_offset[0] + target_extent[0], mqh, map_offset[1] + target_extent[1]])
            layout_features_full = get_local_waymo_map_features(
                layout_context.raw_data, map_transform, bmin, bmax, voxel_size,
                layout_cfg.types, layout_cfg.include_vehicles, max_elements=-1)
            layout_shape = (int(target_extent[0]), mqh, int(target_extent[1]))
            voxelized_layout = vector_to_voxel_map_waymo(
                torch.from_numpy(layout_features_full).to(device),
                layout_shape, factor, layout_cfg, voxel_size, device)
        elif has_layout_cond:
            rx0 = layout_context.region_offset[0] if layout_context.region_offset else 0
            ry0 = layout_context.region_offset[1] if layout_context.region_offset else 0
            bmin = np.array([rx0, 0, ry0], dtype=np.float32)
            bmax = np.array([rx0 + target_extent[0], ocs[1], ry0 + target_extent[1]], dtype=np.float32)
            layout_features_full = get_local_layout_features(
                dataset_type, layout_context.raw_data,
                scene_shape=layout_context.scene_shape, bbox_min=bmin, bbox_max=bmax)
            layout_shape = (int(target_extent[0]), ocs[1], int(target_extent[1]))
            voxelized_layout = voxelize_layout(
                torch.from_numpy(layout_features_full).to(device),
                layout_shape, factor, layout_cfg, voxel_size, device)

        # ---- Step 3: surface heights ----
        chunk_surface_heights: Dict[Tuple[int, int], int] = {}
        if inherited_surface_heights:
            for (x, y), z in inherited_surface_heights.items():
                chunk_surface_heights[(x, y)] = z // factor
        elif layout_features_full is not None and dataset_type == "waymo":
            for p in positions:
                xc, yc = int(p[0]), int(p[1])
                zs = find_lowest_map_chunk_surface(layout_features_full, xc, yc, (ocs[0], ocs[2]), factor=4)
                chunk_surface_heights[(xc, yc)] = zs // factor
        for p in positions:
            chunk_surface_heights.setdefault((int(p[0]), int(p[1])), 0)

        # ---- Step 4: full latent ----
        max_surf_z = max(chunk_surface_heights.values(), default=0)
        total_z = max_surf_z + chunk_z
        if voxelized_layout is not None and voxelized_layout.shape[2] < total_z:
            voxelized_layout = F.pad(voxelized_layout, (0, 0, 0, total_z - voxelized_layout.shape[2], 0, 0), value=0)

        latent_fill = 1.0 if (use_source_flow or is_first_level) else 0.0
        full_latent = torch.full((1, H_channels, target_latent[0], total_z, target_latent[1]),
                                 latent_fill, device=device)
        full_noise = torch.randn_like(full_latent)
        known_mask = torch.zeros(1, target_latent[0], total_z, target_latent[1], device=device)
        surface_mapping: Dict[Tuple[int, int], int] = {}
        has_outpainting = int(unet.config.outpainting_channels) > 0

        # ---- Step 6: generate chunks (simultaneous feather-averaged, or sequential) ----
        if simultaneous:
            chunk_infos = prepare_chunk_infos(
                positions=positions, chunk_surface_heights=chunk_surface_heights,
                full_noise=full_noise, chunk_extent=chunk_extent, chunk_z=chunk_z,
                factor=factor, meta=meta, voxelized_layout=voxelized_layout,
                resolved_tags=resolved_tags, is_first_level=is_first_level,
                coarse_voxels=coarse_voxels, prev_voxel_size=prev_voxel_size,
                truncation=truncation, device=device,
                setup_coarse_conditioning_fn=_setup_coarse_conditioning,
                has_layout_cond=has_layout_cond, has_outpainting=has_outpainting)
            uncovered_fill = 1.0 if (use_source_flow or is_first_level) else 0.0
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else _nullctx()
            with autocast:
                full_latent, surface_mapping, _sim_info = run_simultaneous(
                    unet=unet, scheduler=scheduler, full_shape=tuple(full_latent.shape),
                    full_noise=full_noise, chunk_infos=chunk_infos,
                    chunk_extent_xzy=(chunk_extent[0], chunk_z, chunk_extent[1]),
                    steps=sampling_steps, cfg_scale=cfg_scale, feather_margin=feather_margin,
                    batch_size=simultaneous_batch_size, predict_velocity_fn=self._predict_velocity,
                    uncovered_fill=uncovered_fill, device=device)
        else:
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else _nullctx()
            with autocast:
                for idx, pos in enumerate(tqdm(positions, desc=f"gen {meta.get('level_name')}")):
                    xc, yc = int(pos[0]), int(pos[1])
                    xi, yi = xc // factor, yc // factor
                    zi = chunk_surface_heights.get((xc, yc), 0)
                    sl = (slice(None), slice(None), slice(xi, xi + chunk_extent[0]),
                          slice(zi, zi + chunk_z), slice(yi, yi + chunk_extent[1]))
                    x0 = full_latent[sl].clone()
                    mask = known_mask[:, xi:xi + chunk_extent[0], zi:zi + chunk_z, yi:yi + chunk_extent[1]].clone().unsqueeze(0)
                    chunk_noise = full_noise[sl].clone()

                    cond: Dict[str, torch.Tensor] = {}
                    if has_layout_cond and voxelized_layout is not None:
                        cond["layout"] = get_map_chunk((xi, yi), (chunk_extent[0], chunk_z, chunk_extent[1]),
                                                       voxelized_layout, zi).unsqueeze(0)
                    if has_tags and resolved_tags is not None:
                        cond["tags"] = prepare_tag_conditioning(
                            resolved_tags, (1, H_channels, chunk_extent[0], chunk_z, chunk_extent[1]),
                            device, tag_cfg.types)
                    if has_outpainting:
                        if idx == 0:
                            cond["outpaint_mask"] = torch.ones_like(mask)
                            cond["outpaint_latent"] = torch.zeros_like(x0)
                        else:
                            cond["outpaint_mask"] = 1.0 - mask
                            cond["outpaint_latent"] = x0 * mask

                    coarse_geometry = initial_sample = None
                    if not is_first_level and coarse_voxels is not None:
                        r = _setup_coarse_conditioning(
                            coarse_voxels, meta, xc, yc, zi * factor, chunk_extent, chunk_z, factor,
                            full_noise, xi, yi, device, truncation,
                            fine_voxel_size=voxel_size, coarse_voxel_size=prev_voxel_size)
                        coarse_geometry, initial_sample = r["coarse_geometry"], r["initial_sample"]

                    shape = (1, H_channels, chunk_extent[0], chunk_z, chunk_extent[1])
                    if has_outpainting:
                        patch = self._sample_chunk(unet, scheduler, shape, cond, cfg_scale, sampling_steps,
                                                   coarse_geometry=coarse_geometry, initial_sample=initial_sample)
                    else:
                        patch = self._sample_chunk(unet, scheduler, shape, cond, cfg_scale, sampling_steps,
                                                   coarse_geometry=coarse_geometry, initial_sample=initial_sample,
                                                   mask=mask, x0=x0, noise=chunk_noise)
                    full_latent[sl] = patch
                    known_mask[:, xi:xi + chunk_extent[0], zi:zi + chunk_z, yi:yi + chunk_extent[1]] = 1.0
                    surface_mapping[(xi, yi)] = zi

        # ---- Step 7: reconstruct ----
        voxels = _reconstruct_voxels(
            full_latent, meta, direct_diffusion, truncation,
            coarse_voxels=coarse_voxels, fine_voxel_size=voxel_size,
            coarse_voxel_size=prev_voxel_size if prev_voxel_size else voxel_size)

        return LevelResult(
            voxels=voxels, surface_mapping=surface_mapping, voxel_size=voxel_size,
            factor=factor, map_offset=map_offset, output_chunk_size=ocs,
            map_transform=map_transform, map_extent=map_extent, chunk_boxes=chunk_boxes,
            target_extent=target_extent)

    @torch.no_grad()
    def __call__(self, layout_context: Optional[LayoutContext] = None, scene_extent=None,
                 tags=None, cfg_scale=3.0, sampling_steps=50, overlap_ratio=0.3,
                 smaller_map=False, fraction=0.1, one_chunk=False,
                 use_map_chunking=False, use_uniform_chunking=False,
                 simultaneous=False, simultaneous_batch_size=4, feather_margin=8,
                 refine=False, refine_tags=None, refine_sampling_steps=30,
                 fine_overlap_ratio=0.5, return_coarse=False):
        """Generate a scene: coarse level, then the registered refinement cascade."""
        sim = dict(simultaneous=simultaneous, simultaneous_batch_size=simultaneous_batch_size,
                   feather_margin=feather_margin)
        coarse = self._generate_level(
            self.unet, self.scheduler, layout_context=layout_context, scene_extent=scene_extent,
            tags=tags, cfg_scale=cfg_scale, sampling_steps=sampling_steps,
            overlap_ratio=overlap_ratio, smaller_map=smaller_map, fraction=fraction,
            one_chunk=one_chunk, **sim)

        if not refine or not self._refinements:
            return coarse

        prev = coarse
        for unet, scheduler, _vae in self._refinements:
            result = self._generate_level(
                unet, scheduler, layout_context=layout_context,
                tags=refine_tags if refine_tags is not None else tags,
                cfg_scale=cfg_scale, sampling_steps=refine_sampling_steps,
                overlap_ratio=fine_overlap_ratio, smaller_map=smaller_map, fraction=fraction,
                coarse_voxels=prev.voxels, prev_surface_mapping=prev.surface_mapping,
                prev_voxel_size=prev.voxel_size, prev_chunk_size=prev.output_chunk_size,
                prev_factor=prev.factor, prev_map_offset=prev.map_offset,
                use_map_chunking=use_map_chunking, use_uniform_chunking=use_uniform_chunking,
                one_chunk=one_chunk, **sim)
            prev = result

        if return_coarse:
            return prev, coarse
        return prev


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False

"""Simultaneous (feather-averaged) multi-chunk flow sampler — diffusers-native.

Faithful, self-contained port of the research simultaneous sampler used by
``ScenePipeline`` for ``--simultaneous`` generation. Instead of generating chunks
sequentially with inpainting overlap, every chunk is integrated through the flow
ODE at the same time; at each Euler step the per-chunk velocity predictions are
feather-weight-averaged in the overlap regions and a single Euler update is
applied to the full volume.

Ported from:
  * ``src/models/schedulers/flow_euler_simultaneous_sampler.py``
    (``ChunkInfo``, ``_build_feather_weight``, ``_stack_cond_dicts``,
    ``sample_simultaneous``)
  * ``src/inference/scene_pipeline.py``
    (``_prepare_all_chunk_infos`` lines 508-603,
    ``_run_simultaneous_generation`` lines 604-667)

This module is callback-driven so it stays decoupled from the pipeline:
  * ``predict_velocity_fn`` — supplied by ``WorldFlow3DPipeline`` so CFG matches
    ``WorldFlow3DPipeline._predict_velocity`` *exactly* (zero only ``cond["layout"]``
    for the unconditional half, return ``v_uncond + cfg*(v_cond - v_uncond)``,
    model time = ``1000 * t``).
  * ``setup_coarse_conditioning_fn`` — supplied by the pipeline so coarse-geometry
    / source-flow ``initial_sample`` extraction matches the sequential path.

All tensors follow the project axis order ``[B, C, X, Z, Y]`` (Z is vertical).
The full volume + accumulators live on CPU; only per-chunk batches move to GPU
for the forward passes (CPU off-loading), matching the research sampler.

References:
  * MultiDiffusion (Bar-Tal et al., 2023)
  * Flow Matching for Generative Modeling (Lipman et al., 2022)
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..conditioning.tags import prepare_tag_conditioning
from ..conditioning.waymo import get_map_chunk

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x


# ---------------------------------------------------------------------------
# Per-chunk record (ported verbatim from flow_euler_simultaneous_sampler.py)
# ---------------------------------------------------------------------------

@dataclass
class ChunkInfo:
    """Pre-computed information for a single chunk.

    All spatial indices are in latent space.

    Attributes:
        index: Position in the chunk list (for tracking).
        xi: X start in full latent (latent coords).
        yi: Y start in full latent (latent coords).
        zi: Z start in full latent (latent coords).
        cond: Dict of conditioning tensors (layout, tags, outpaint_*, etc.),
            each with batch dim 1.
        coarse_geometry: Optional coarse geometry tensor ``[1, C_geo, ...]``.
        initial_sample: Optional initial_sample for source-flow models
            ``[1, H_channels, cx, cz, cy]``.
    """
    index: int
    xi: int
    yi: int
    zi: int
    cond: Dict[str, torch.Tensor] = field(default_factory=dict)
    coarse_geometry: Optional[torch.Tensor] = None
    initial_sample: Optional[torch.Tensor] = None


def _stack_cond_dicts(cond_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack per-chunk conditioning dicts along the batch dimension.

    Each dict has tensors with batch=1. Concatenates to batch=N. Keys present in
    some dicts but absent in others are zero-padded (using a present reference).
    """
    if not cond_list:
        return {}
    all_keys = set()
    for c in cond_list:
        all_keys.update(c.keys())
    result: Dict[str, torch.Tensor] = {}
    for k in all_keys:
        tensors = []
        for c in cond_list:
            if k in c:
                tensors.append(c[k])
            else:
                ref = next(c2[k] for c2 in cond_list if k in c2)
                tensors.append(torch.zeros_like(ref))
        result[k] = torch.cat(tensors, dim=0)
    return result


def _build_feather_weight(
    cx: int, cz: int, cy: int,
    margin: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a ``[1, 1, cx, cz, cy]`` feathering weight tensor.

    Ramps linearly from ``1/(margin+1)`` at boundaries to ``1.0`` in the interior.
    Uses ``min`` across dimensions at corners so corner voxels get the lowest ramp
    value rather than the product.

    Args:
        cx: Chunk size in X.
        cz: Chunk size in Z.
        cy: Chunk size in Y.
        margin: Number of voxels over which to ramp weights.
        device: Torch device.
        dtype: Tensor dtype.

    Returns:
        Feather weight tensor of shape ``[1, 1, cx, cz, cy]``.
    """
    if margin <= 0:
        return torch.ones(1, 1, cx, cz, cy, device=device, dtype=dtype)

    def _ramp(length: int) -> torch.Tensor:
        w = torch.ones(length, device=device, dtype=dtype)
        m = min(margin, length // 2)
        for i in range(m):
            v = (i + 1) / (m + 1)
            w[i] = min(w[i], v)
            w[length - 1 - i] = min(w[length - 1 - i], v)
        return w

    wx = _ramp(cx)
    wz = _ramp(cz)
    wy = _ramp(cy)
    weight = torch.min(wx[:, None, None], torch.min(wz[None, :, None], wy[None, None, :]))
    return weight.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Chunk-info preparation (port of ScenePipeline._prepare_all_chunk_infos)
# ---------------------------------------------------------------------------

def prepare_chunk_infos(
    positions: List[Tuple[float, float]],
    chunk_surface_heights: Dict[Tuple[int, int], int],
    full_noise: torch.Tensor,
    chunk_extent: Tuple[int, int],
    chunk_z: int,
    factor: int,
    meta: Dict[str, Any],
    voxelized_layout: Optional[torch.Tensor],
    resolved_tags: Optional[Dict[str, str]],
    is_first_level: bool,
    coarse_voxels: Optional[torch.Tensor],
    prev_voxel_size: Optional[float],
    truncation: float,
    device: torch.device,
    setup_coarse_conditioning_fn: Callable,
    has_layout_cond: bool = True,
    has_outpainting: bool = False,
    offload_to_cpu: bool = True,
) -> List[ChunkInfo]:
    """Pre-compute conditioning for all chunks (for simultaneous generation).

    Extracts the same per-chunk information as the sequential tiling loop in
    ``WorldFlow3DPipeline._generate_level`` (Step 6 cond build + coarse setup),
    but returns a list of :class:`ChunkInfo` objects instead of sampling.

    Args:
        positions: List of ``(x, y)`` chunk top-left positions in finest voxels.
        chunk_surface_heights: ``{(x, y): zi}`` per-chunk latent z-start.
        full_noise: ``[1, H, X_lat, Z_lat, Y_lat]`` shared noise volume; passed to
            ``setup_coarse_conditioning_fn`` for source-flow color-channel noise.
        chunk_extent: ``(cx, cy)`` chunk dims in *latent* space (X, Y).
        chunk_z: Chunk size in Z (latent space).
        factor: Latent downscale factor (1 for direct-diffusion).
        meta: ``unet.config.level_meta`` dict.
        voxelized_layout: Full voxelized layout volume, or ``None``.
        resolved_tags: Resolved tag dict, or ``None``.
        is_first_level: True for the coarse level (no coarse geometry).
        coarse_voxels: Previous-level voxels, or ``None``.
        prev_voxel_size: Previous-level voxel size (for coarse cropping).
        truncation: UDF truncation value.
        device: Torch device.
        setup_coarse_conditioning_fn: Callback matching the package's
            ``_setup_coarse_conditioning(coarse_voxels, meta, x_coord, y_coord,
            z_offset, chunk_extent, chunk_z, factor, full_noise, xi, yi, device,
            truncation, fine_voxel_size, coarse_voxel_size) -> dict`` returning
            ``{"coarse_geometry": ..., "initial_sample": ...}``.
        has_layout_cond: Whether layout conditioning is active.
        has_outpainting: Whether the model takes outpainting channels.
        offload_to_cpu: Move coarse_geometry / initial_sample to CPU immediately
            after computation to avoid accumulating GPU memory across many chunks.

    Returns:
        List of :class:`ChunkInfo` (one per position).
    """
    H_channels = int(meta["H_channels"])
    has_tags = bool(meta.get("has_tags"))
    tag_types = meta.get("tag_types") or []
    voxel_size = float(meta["voxel_size"])
    cx, cy = chunk_extent

    infos: List[ChunkInfo] = []
    for idx, pos in enumerate(positions):
        x_coord, y_coord = int(pos[0]), int(pos[1])
        xi, yi = x_coord // factor, y_coord // factor
        zi = chunk_surface_heights.get((x_coord, y_coord), 0)

        cond: Dict[str, torch.Tensor] = {}

        # Layout conditioning — store on CPU
        if has_layout_cond and voxelized_layout is not None:
            cond["layout"] = get_map_chunk(
                (xi, yi), (cx, chunk_z, cy), voxelized_layout, zi
            ).unsqueeze(0).cpu()

        # Tag conditioning — store on CPU
        if has_tags and resolved_tags is not None:
            chunk_shape = (1, H_channels, cx, chunk_z, cy)
            tag_cond = prepare_tag_conditioning(
                resolved_tags, chunk_shape, device, tag_types,
            )
            cond["tags"] = tag_cond.cpu()

        # Outpainting conditioning — create directly on CPU
        if has_outpainting:
            cond["outpaint_mask"] = torch.ones(1, 1, cx, chunk_z, cy)
            cond["outpaint_latent"] = torch.zeros(1, H_channels, cx, chunk_z, cy)

        # Coarse-geometry / source-flow conditioning (fine levels only)
        coarse_geometry = None
        initial_sample = None
        if not is_first_level and coarse_voxels is not None:
            cond_result = setup_coarse_conditioning_fn(
                coarse_voxels, meta, x_coord, y_coord, zi * factor,
                chunk_extent, chunk_z, factor,
                full_noise, xi, yi,
                device, truncation,
                fine_voxel_size=voxel_size,
                coarse_voxel_size=prev_voxel_size,
            )
            coarse_geometry = cond_result.get("coarse_geometry")
            initial_sample = cond_result.get("initial_sample")
            if offload_to_cpu:
                if coarse_geometry is not None:
                    coarse_geometry = coarse_geometry.cpu()
                if initial_sample is not None:
                    initial_sample = initial_sample.cpu()

        infos.append(ChunkInfo(
            index=idx,
            xi=xi, yi=yi, zi=zi,
            cond=cond,
            coarse_geometry=coarse_geometry,
            initial_sample=initial_sample,
        ))

    return infos


# ---------------------------------------------------------------------------
# Core simultaneous sampling (port of sample_simultaneous + run wrapper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_simultaneous(
    unet,
    scheduler,
    full_shape: Tuple[int, ...],
    full_noise: Optional[torch.Tensor],
    chunk_infos: List[ChunkInfo],
    chunk_extent_xzy: Tuple[int, int, int],
    steps: int,
    cfg_scale: float,
    feather_margin: int,
    batch_size: int,
    predict_velocity_fn: Callable,
    uncovered_fill: float = 1.0,
    temperature: float = 1.0,
    rescale_t: float = 1.0,
    verbose: bool = False,
    store_trajectory: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[Tuple[int, int], int], Dict[str, Any]]:
    """Generate all chunks simultaneously with feather-weighted velocity averaging.

    At each ODE timestep:
      1. Extract each chunk's state from the (CPU) full volume.
      2. Predict velocity per chunk, batched for GPU efficiency, via
         ``predict_velocity_fn`` (CFG handled there — must match the sequential
         pipeline's ``_predict_velocity``).
      3. Accumulate feather-weighted velocities into a CPU accumulator.
      4. Coverage-normalize and apply one Euler update: ``full -= dt * vel_avg``.

    CPU off-loading: the full volume, all accumulators and the coverage map live
    on CPU; only the per-chunk state / conditioning batches are moved to ``device``
    for the forward passes.

    Args:
        unet: The level UNet (used for the forward + to infer ``device``).
        scheduler: ``WorldFlow3DFlowScheduler`` (used for the ``rescale_t`` schedule
            via the same ``1 -> 0`` reparameterization; the Euler step is applied
            inline here so velocity averaging happens before the update).
        full_shape: ``(1, H_channels, X_lat, Z_lat, Y_lat)`` full volume shape.
        full_noise: Optional ``[1, H, X, Z, Y]`` shared noise for coherent
            initialization. If ``None``, random noise is generated.
        chunk_infos: List of :class:`ChunkInfo` with pre-computed conditioning.
        chunk_extent_xzy: ``(cx, cz, cy)`` chunk dims in latent space.
        steps: Number of Euler integration steps.
        cfg_scale: Classifier-free guidance scale (passed to ``predict_velocity_fn``).
        feather_margin: Voxels over which to ramp feather weights at chunk
            boundaries. ``0`` = equal-weight averaging.
        batch_size: Max chunks per batched forward pass (CFG doubles this inside
            ``predict_velocity_fn``).
        predict_velocity_fn: ``fn(unet, x, t, cond, cfg_scale, coarse_geometry)
            -> velocity``. Must zero only ``cond["layout"]`` for the uncond half
            and return ``v_uncond + cfg*(v_cond - v_uncond)``; model time = 1000*t.
        uncovered_fill: Value to fill regions covered by no chunk (post-sampling).
            ``1.0`` (= truncation after denorm) for source-flow / coarsest levels;
            ``0.0`` for standard residual denoising.
        temperature: Noise temperature for initialization.
        rescale_t: Time rescaling factor (matches the sequential scheduler).
        verbose: Show a progress bar.
        store_trajectory: Store per-step full-volume snapshots in the info dict.
        device: Target device for forward passes. Defaults to ``unet``'s device.

    Returns:
        full_latent: ``(1, H_channels, X_lat, Z_lat, Y_lat)`` final volume, on the
            same device as ``full_noise`` (or ``device`` if ``full_noise`` is None).
        surface_mapping: ``{(xi, yi): zi}`` for downstream reconstruction.
        info: Dict with optional ``'trajectory'`` list and ``'coverage'`` tensor.
    """
    if device is None:
        try:
            device = next(unet.parameters()).device
        except StopIteration:  # pragma: no cover
            device = torch.device("cpu")

    cx, cz, cy = chunk_extent_xzy
    N = len(chunk_infos)
    out_device = full_noise.device if full_noise is not None else device

    # ---- 1. Initialize full volume on CPU to save GPU memory ----
    if full_noise is not None:
        full_sample = full_noise.detach().cpu().clone() * temperature
    else:
        full_sample = torch.randn(full_shape) * temperature

    # Feather weights: CPU copy for accumulation (GPU copy is unused here since
    # velocity weighting also happens on CPU, mirroring the research sampler).
    feather_cpu = _build_feather_weight(cx, cz, cy, feather_margin, torch.device("cpu"))

    # ---- 2. Blend initial_samples for source-flow models (on CPU) ----
    has_initial_samples = any(ci.initial_sample is not None for ci in chunk_infos)
    if has_initial_samples:
        init_accum = torch.zeros(full_shape)
        init_weight = torch.zeros(1, 1, *full_shape[2:])
        for ci in chunk_infos:
            if ci.initial_sample is not None:
                init_accum[:, :, ci.xi:ci.xi + cx, ci.zi:ci.zi + cz, ci.yi:ci.yi + cy] += (
                    ci.initial_sample.detach().cpu() * feather_cpu
                )
                init_weight[:, :, ci.xi:ci.xi + cx, ci.zi:ci.zi + cz, ci.yi:ci.yi + cy] += feather_cpu
        covered = init_weight > 0
        full_sample = torch.where(
            covered.expand_as(full_sample),
            init_accum / init_weight.clamp(min=1e-8),
            full_sample,
        )

    # ---- 3. Pre-compute coverage map on CPU ----
    coverage = torch.zeros(1, 1, *full_shape[2:])
    for ci in chunk_infos:
        coverage[:, :, ci.xi:ci.xi + cx, ci.zi:ci.zi + cz, ci.yi:ci.yi + cy] += feather_cpu

    # ---- 4. Time schedule (matches scheduler.set_timesteps rescale_t) ----
    t_seq = np.linspace(1, 0, steps + 1)
    if rescale_t != 1.0:
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

    trajectory: List[torch.Tensor] = []

    # Ensure chunk conditioning stays on CPU until needed.
    for ci in chunk_infos:
        ci.cond = {k: v.detach().cpu() for k, v in ci.cond.items()}
        if ci.coarse_geometry is not None:
            ci.coarse_geometry = ci.coarse_geometry.detach().cpu()

    # ---- 5. Main sampling loop ----
    iterator = tqdm(t_pairs, desc="Simultaneous Flow Sampling", disable=not verbose)
    for t, t_prev in iterator:
        t = float(t)
        dt = t - t_prev

        # Accumulate feather-weighted velocities on CPU.
        vel_accum = torch.zeros(full_shape)

        for b_start in range(0, N, batch_size):
            b_end = min(b_start + batch_size, N)
            batch_cis = chunk_infos[b_start:b_end]

            # Extract chunk states from CPU volume, move to GPU.
            x_t_patches = []
            for ci in batch_cis:
                patch = full_sample[:, :, ci.xi:ci.xi + cx, ci.zi:ci.zi + cz, ci.yi:ci.yi + cy]
                x_t_patches.append(patch.clone().to(device))
            x_t_batch = torch.cat(x_t_patches, dim=0)

            # Stack conditioning dicts and move to GPU.
            cond_batch = _stack_cond_dicts([ci.cond for ci in batch_cis])
            cond_batch = {k: v.to(device) for k, v in cond_batch.items()}
            if not cond_batch:
                cond_batch = None

            # Stack coarse geometry and move to GPU.
            coarse_batch = None
            if batch_cis[0].coarse_geometry is not None:
                coarse_batch = torch.cat(
                    [ci.coarse_geometry for ci in batch_cis], dim=0
                ).to(device)

            # Predict velocity (CFG handled by callback to match the pipeline).
            v_batch = predict_velocity_fn(
                unet, x_t_batch, t, cond_batch, cfg_scale, coarse_batch,
            )

            # Scatter feather-weighted velocities back to the CPU accumulator.
            v_batch_cpu = v_batch.float().cpu()
            for j, ci in enumerate(batch_cis):
                vel_accum[:, :, ci.xi:ci.xi + cx, ci.zi:ci.zi + cz, ci.yi:ci.yi + cy] += (
                    v_batch_cpu[j:j + 1] * feather_cpu
                )

            # Free GPU intermediates (let the caching allocator reuse the blocks;
            # empty_cache() here would force a pool release + sync every batch, and
            # the next same-shape batch just re-requests identical blocks).
            del x_t_batch, cond_batch, coarse_batch, v_batch, v_batch_cpu

        # Average velocity where covered, update full volume (all on CPU).
        cov_exp = coverage.expand_as(vel_accum)
        vel_avg = torch.where(
            cov_exp > 0,
            vel_accum / cov_exp.clamp(min=1e-8),
            torch.zeros_like(vel_accum),
        )
        full_sample = full_sample - dt * vel_avg

        if store_trajectory:
            trajectory.append(full_sample.clone())

    # ---- 6. Move back to the original device for reconstruction ----
    full_latent = full_sample.to(out_device)

    # ---- 7. Fill regions covered by no chunk (pure noise) with uncovered_fill ----
    binary_cov = torch.zeros(1, 1, *full_latent.shape[2:], device=full_latent.device)
    for ci in chunk_infos:
        binary_cov[:, :, ci.xi:ci.xi + cx, ci.zi:ci.zi + cz, ci.yi:ci.yi + cy] = 1.0
    uncovered = (binary_cov == 0).expand_as(full_latent)
    full_latent[uncovered] = uncovered_fill

    # ---- 8. Surface mapping for downstream levels ----
    surface_mapping: Dict[Tuple[int, int], int] = {}
    for ci in chunk_infos:
        surface_mapping[(ci.xi, ci.yi)] = ci.zi

    info: Dict[str, Any] = {"trajectory": trajectory, "coverage": coverage}
    return full_latent, surface_mapping, info

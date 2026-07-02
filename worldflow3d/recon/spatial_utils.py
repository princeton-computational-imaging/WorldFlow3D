# Surgically extracted from src/utils/spatial_utils.py (the 3 functions the
# pipeline needs). The source module imports cv2/numba/matplotlib at module
# level, so it cannot be whole-file vendored; keep these in sync by hand.
"""Spatial helpers: surface-height lookup and surface-mapping propagation."""
import math
from typing import Dict, Tuple

import numpy as np


def find_lowest_map_chunk_surface(map_features: np.ndarray, x: int, y: int,
                                  chunk_size: Tuple[int, int], factor: int = 4) -> int:
    """Find lowest surface Z from raw map features within a chunk's XY extent.

    Args:
        map_features: Raw map features [N, 35] with positions in [:, 0:3] as (x, z, y)
        x, y: Chunk position in output voxel space
        chunk_size: (chunk_x, chunk_y) in output voxel space
        factor: Alignment factor for z-offset

    Returns:
        Lowest z-offset aligned to factor, or 0 if no map content
    """
    if len(map_features) == 0:
        return 0

    positions = map_features[:, :3]
    x_end = x + chunk_size[0]
    y_end = y + chunk_size[1]

    in_chunk = (
        (positions[:, 0] >= x) & (positions[:, 0] < x_end) &
        (positions[:, 2] >= y) & (positions[:, 2] < y_end)
    )

    if not in_chunk.any():
        return 0

    z_values = positions[in_chunk, 1]
    lowest_z = z_values.min()
    lowest_z_aligned = (int(lowest_z) // factor) * factor
    return max(0, lowest_z_aligned)


def transform_surface_mapping_positions(
    surface_mapping: Dict[Tuple[int, int], int],
    prev_voxel_size: float,
    prev_chunk_size: Tuple[int, int, int],
    new_voxel_size: float,
    new_chunk_size: Tuple[int, int, int],
    overlap_ratio: float = 0.5,
) -> Dict[Tuple[int, int], int]:
    """Transform surface map from one level to the next, with subdivision or merging.

    See src/utils/spatial_utils.py for the full docstring. Positions are in
    output (voxel-grid) space; Z values propagate from parent to subdivided
    children. Handles subdivision / 1:1 / merge.
    """
    prev_world_size_x = prev_chunk_size[0] * prev_voxel_size
    prev_world_size_y = prev_chunk_size[2] * prev_voxel_size

    new_world_size_x = new_chunk_size[0] * new_voxel_size
    new_world_size_y = new_chunk_size[2] * new_voxel_size

    subdiv_x = max(1, int(round(prev_world_size_x / new_world_size_x)))
    subdiv_y = max(1, int(round(prev_world_size_y / new_world_size_y)))

    stride_x = max(1, int(new_chunk_size[0] * (1.0 - overlap_ratio)))
    stride_y = max(1, int(new_chunk_size[2] * (1.0 - overlap_ratio)))

    if overlap_ratio > 0 and subdiv_x > 1:
        total_width = subdiv_x * new_chunk_size[0]
        subdiv_x = max(1, (total_width - new_chunk_size[0]) // stride_x + 1)
        if (subdiv_x - 1) * stride_x + new_chunk_size[0] < total_width:
            subdiv_x += 1
    if overlap_ratio > 0 and subdiv_y > 1:
        total_height = subdiv_y * new_chunk_size[2]
        subdiv_y = max(1, (total_height - new_chunk_size[2]) // stride_y + 1)
        if (subdiv_y - 1) * stride_y + new_chunk_size[2] < total_height:
            subdiv_y += 1

    coord_scale = prev_voxel_size / new_voxel_size

    new_surface_mapping = {}

    for (x_prev, y_prev), z_prev in surface_mapping.items():
        x_world = x_prev * prev_voxel_size
        y_world = y_prev * prev_voxel_size

        x_new = x_world / new_voxel_size
        y_new = y_world / new_voxel_size

        z_new = int(z_prev * coord_scale)

        if subdiv_x == 1 and subdiv_y == 1:
            new_surface_mapping[(int(round(x_new)), int(round(y_new)))] = z_new
        else:
            for i in range(subdiv_x):
                for j in range(subdiv_y):
                    x_sub = int(round(x_new)) + i * stride_x
                    y_sub = int(round(y_new)) + j * stride_y
                    new_surface_mapping[(x_sub, y_sub)] = z_new

    return new_surface_mapping


def sample_finer_surface_mapping_positions(
    surface_mapping: Dict[Tuple[int, int], int],
    prev_voxel_size: float,
    prev_chunk_size: Tuple[int, int, int],
    new_voxel_size: float,
    new_chunk_size: Tuple[int, int, int],
    overlap_ratio: float = 0.5,
) -> Dict[Tuple[int, int], int]:
    """Sample new chunk positions on a uniform grid over the merged previous area.

    See src/utils/spatial_utils.py for the full docstring. Guarantees uniform
    overlap and full coverage of the previous area; Z from the lowest
    overlapping previous chunk.
    """
    if not surface_mapping:
        return {}

    coord_scale = prev_voxel_size / new_voxel_size
    cx_new = new_chunk_size[0]
    cy_new = new_chunk_size[2]
    cx_prev = prev_chunk_size[0]
    cy_prev = prev_chunk_size[2]

    prev_rects = []  # (x_min, y_min, x_max, y_max, z_new)
    for (x_prev, y_prev), z_prev in surface_mapping.items():
        x_new = x_prev * coord_scale
        y_new = y_prev * coord_scale
        z_new = int(z_prev * coord_scale)
        x_end = x_new + cx_prev * coord_scale
        y_end = y_new + cy_prev * coord_scale
        prev_rects.append((x_new, y_new, x_end, y_end, z_new))

    min_x = min(r[0] for r in prev_rects)
    min_y = min(r[1] for r in prev_rects)
    max_x = max(r[2] for r in prev_rects)
    max_y = max(r[3] for r in prev_rects)

    stride_x = max(1, int(round(cx_new * (1.0 - overlap_ratio))))
    stride_y = max(1, int(round(cy_new * (1.0 - overlap_ratio))))

    area_x = max_x - min_x
    area_y = max_y - min_y

    if area_x <= cx_new:
        n_x = 1
    else:
        n_x = math.ceil((area_x - cx_new) / stride_x) + 1

    if area_y <= cy_new:
        n_y = 1
    else:
        n_y = math.ceil((area_y - cy_new) / stride_y) + 1

    start_x = int(round(min_x))
    start_y = int(round(min_y))

    new_surface_mapping: Dict[Tuple[int, int], int] = {}

    for i in range(n_x):
        for j in range(n_y):
            x_pos = start_x + i * stride_x
            y_pos = start_y + j * stride_y

            z_val = None
            for (px_min, py_min, px_max, py_max, pz) in prev_rects:
                if (x_pos < px_max and x_pos + cx_new > px_min and
                        y_pos < py_max and y_pos + cy_new > py_min):
                    if z_val is None:
                        z_val = pz
                    else:
                        z_val = min(z_val, pz)

            if z_val is None:
                z_val = 0

            new_surface_mapping[(x_pos, y_pos)] = z_val

    verified: Dict[Tuple[int, int], int] = {}
    for (x_pos, y_pos), z_val in new_surface_mapping.items():
        cx_center = x_pos + cx_new / 2.0
        cy_center = y_pos + cy_new / 2.0
        for (px_min, py_min, px_max, py_max, _pz) in prev_rects:
            if px_min <= cx_center <= px_max and py_min <= cy_center <= py_max:
                verified[(x_pos, y_pos)] = z_val
                break

    return verified

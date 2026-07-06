# Surgically extracted from src/utils/mesh_utils.py (dense-path functions only;
# the source module also has PCA/foundation helpers we don't ship). Keep in sync
# by hand. get_surface_threshold is inlined from src/utils/common_utils.py.
"""Voxel volume -> mesh (marching cubes) + colored .ply export."""
import numpy as np
import torch
import trimesh
from scipy.ndimage import map_coordinates
from skimage import measure


def get_surface_threshold(voxel_size: float) -> float:
    return voxel_size * 0.8


def truncate_udf(voxels: torch.Tensor, truncation: float) -> torch.Tensor:
    """Clamp UDF values to [0, truncation]. ``truncation < 0`` => unchanged."""
    if truncation < 0:
        return voxels
    return torch.clamp(voxels, min=0, max=truncation)


def interpolate_vertex_colors(vertices, color_volume, order=0):
    """Interpolate colors at vertex positions.

    Args:
        vertices: [N, 3] vertex positions in voxel coordinates [X, Z, Y]
        color_volume: [X, Z, Y, 3] color volume in range [0, 1]
        order: interpolation order (0=nearest, 1=linear)

    Returns:
        colors: [N, 3] RGB colors (uint8)
    """
    has_color_mask = np.any(color_volume > 0, axis=-1)  # [X, Z, Y]
    sparsity = 1.0 - (has_color_mask.sum() / has_color_mask.size)

    if sparsity > 0.5:  # sparse volume -> nearest non-zero via KDTree
        from scipy.spatial import cKDTree

        nonzero_coords = np.argwhere(has_color_mask)
        if len(nonzero_coords) == 0:
            return np.zeros((vertices.shape[0], 3), dtype=np.uint8)
        tree = cKDTree(nonzero_coords.astype(np.float32))
        _, nearest_indices = tree.query(vertices, k=1)
        nearest_coords = nonzero_coords[nearest_indices]
        colors_float = color_volume[nearest_coords[:, 0], nearest_coords[:, 1], nearest_coords[:, 2]]
    else:
        color_volume_t = color_volume.transpose(3, 0, 1, 2)
        coords = vertices.T
        colors_float = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        for c in range(3):
            colors_float[:, c] = map_coordinates(color_volume_t[c], coords, order=order, mode="nearest")

    return (colors_float * 255.0).clip(0, 255).astype(np.uint8)


def volume_to_mesh(volume, threshold=0.0, normalize=True):
    """Convert a (geometry [+color]) volume to a trimesh via marching cubes.

    Args:
        volume: tensor/array, one of
            - [X, Y, Z]            geometry only (Y vertical)
            - [1, X, Y, Z]         geometry (squeezed to first channel)
            - [>=4, X, Y, Z]       geometry + RGB (channels 1:4 in [0, 1])
        threshold: isosurface level
        normalize: normalize vertices to [-1, 1]
    """
    volume = volume.cpu().numpy() if isinstance(volume, torch.Tensor) else volume

    if volume.ndim == 4:
        has_color = volume.shape[0] >= 4
        geom = volume[0]
        if has_color:
            color = volume[1:4].transpose(1, 2, 3, 0)  # [X,Y,Z,3] in [0,1]
    elif volume.ndim == 3:
        has_color = False
        geom = volume
    else:
        raise ValueError(f"Volume shape {volume.shape} not supported")

    vertices, triangles, _, _ = measure.marching_cubes(
        geom, level=threshold, gradient_direction="descent"
    )

    if normalize:
        if len(vertices) > 1:
            vertices = vertices / (geom.shape[1] - 1.0)
            vertices = vertices * 2 - 1.0

    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    if has_color:
        surface_mask = geom <= threshold
        color_masked = color.copy()
        color_masked[~surface_mask] = 0.0
        mesh.visual.vertex_colors = interpolate_vertex_colors(
            mesh.vertices if normalize else vertices, color_masked, order=0
        )

    return mesh


def save_mesh(voxels, voxel_size: float, path: str, offset=None, crop_top: float = 0.0):
    """Save geometry (and optional color) mesh from a voxel volume.

    Args:
        voxels: (1, C, X, Z, Y) tensor. C=1 geometry only; C>=4 geometry+color.
        voxel_size: physical size of each voxel in metres.
        path: output .ply path (color mesh gets a ``_color`` suffix).
        offset: optional (3,) world-space offset subtracted after scaling.
        crop_top: fraction of the top (vertical) extent to drop.
    """
    threshold = get_surface_threshold(voxel_size)
    has_color = voxels.shape[1] >= 4

    geom = voxels[0, 0].cpu().numpy()
    if crop_top > 0.0:
        geom = geom[:, :int(geom.shape[1] * (1.0 - crop_top)), :]

    mesh = volume_to_mesh(geom, threshold=threshold, normalize=False)
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]  # rotate to z-up
    mesh.vertices = mesh.vertices * voxel_size
    if offset is not None:
        mesh.vertices = mesh.vertices - np.asarray(offset)
    mesh.export(path)

    if has_color:
        color_path = str(path).replace(".ply", "_color.ply")
        color_mesh = volume_to_mesh(voxels[0].cpu(), threshold=threshold, normalize=False)
        color_mesh.vertices = color_mesh.vertices[:, [0, 2, 1]]
        color_mesh.vertices = color_mesh.vertices * voxel_size
        if offset is not None:
            color_mesh.vertices = color_mesh.vertices - np.asarray(offset)
        color_mesh.export(color_path)

    return mesh

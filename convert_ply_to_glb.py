#!/usr/bin/env python3
"""Convert all .ply files in assets/meshes/ to .glb format with optional decimation."""

import sys
from pathlib import Path

import trimesh
import numpy as np


def ply_to_glb(ply_path: Path, max_faces: int = 500_000) -> Path:
    glb_path = ply_path.with_suffix(".glb")
    print(f"Loading {ply_path.name} ...")
    mesh = trimesh.load(ply_path)

    if isinstance(mesh, trimesh.Scene):
        print(f"  Scene with {len(mesh.geometry)} geometries")
    else:
        print(f"  Vertices: {len(mesh.vertices):,}  Faces: {len(mesh.faces):,}")

        # Decimate if too many faces
        if len(mesh.faces) > max_faces:
            import fast_simplification
            ratio = 1.0 - (max_faces / len(mesh.faces))
            print(f"  Decimating (reduction {ratio:.3f}) to ~{max_faces:,} faces ...")
            points, tris = fast_simplification.simplify(
                mesh.vertices.astype(np.float32),
                mesh.faces,
                target_reduction=ratio,
            )
            # Preserve vertex colors by nearest-vertex lookup
            if mesh.visual and hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                from scipy.spatial import cKDTree
                tree = cKDTree(mesh.vertices)
                _, idx = tree.query(points)
                colors = mesh.visual.vertex_colors[idx]
                mesh = trimesh.Trimesh(vertices=points, faces=tris, vertex_colors=colors)
            else:
                mesh = trimesh.Trimesh(vertices=points, faces=tris)
            print(f"  After decimation: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

        mesh = trimesh.Scene(mesh)

    mesh.export(str(glb_path), file_type="glb")
    size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"  -> {glb_path.name} ({size_mb:.1f} MB)")
    return glb_path


if __name__ == "__main__":
    meshes_dir = Path(__file__).parent / "assets" / "meshes"
    ply_files = sorted(meshes_dir.glob("*.ply"))

    if not ply_files:
        print("No .ply files found in assets/meshes/")
        sys.exit(1)

    print(f"Found {len(ply_files)} PLY file(s)\n")
    for ply in ply_files:
        ply_to_glb(ply)
    print("\nDone.")

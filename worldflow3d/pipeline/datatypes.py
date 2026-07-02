"""Lightweight data classes for the pipeline (decoupled from the research repo)."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class LayoutContext:
    """Dataset-agnostic conditioning context passed to the pipeline.

    Streets (e.g. Waymo): dataset_type="waymo", raw_data=map JSON, transform_data=coord dict.
    Front3D/ScanNet++:    dataset_type="front3d"|"scannetpp", raw_data=layout dict,
                          scene_shape=(Z,H,X), region_offset=optional (x,y).
    """
    dataset_type: str
    source_id: Optional[str] = None
    raw_data: Optional[dict] = None
    transform_data: Optional[dict] = None
    scene_shape: Optional[Tuple[int, int, int]] = None
    region_offset: Optional[Tuple[int, int]] = None


@dataclass
class LevelResult:
    """Output of one generation level."""
    voxels: torch.Tensor
    surface_mapping: Dict[Tuple[int, int], int]
    voxel_size: float
    factor: int = 1
    map_offset: Tuple[float, float] = (0.0, 0.0)
    output_chunk_size: Optional[Tuple[int, int, int]] = None
    map_transform: Optional[np.ndarray] = None
    map_extent: Optional[list] = None
    chunk_boxes: Optional[list] = None
    target_extent: Optional[list] = None
    info: Optional[dict] = None

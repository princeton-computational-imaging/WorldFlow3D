"""Checkpoint conversion: research checkpoint dir -> diffusers layout.

This subpackage is a **release-time tool**, not part of the runtime inference
path. It runs inside the research environment (so it may import ``src.*`` to read
the config and build an architecture-faithful generator); the produced diffusers
artifacts load with only the standalone ``worldflow3d`` package.
"""
from .convert_checkpoint import convert_level

__all__ = ["convert_level"]

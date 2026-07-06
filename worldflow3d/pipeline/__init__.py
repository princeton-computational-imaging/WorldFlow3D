"""Diffusers-native classes for WorldFlow3D (the diffusers integration layer).

Named ``pipeline`` rather than ``diffusers`` to avoid shadowing the real
``diffusers`` library (notably under ``trust_remote_code`` loading).
"""
from .scheduler import WorldFlow3DFlowScheduler

__all__ = ["WorldFlow3DFlowScheduler"]

# UNet wrapper and the pipeline land later in the build order; export when present.
try:  # pragma: no cover - incremental build
    from .unet import WorldFlow3DUNet  # noqa: F401
    __all__.append("WorldFlow3DUNet")
except ImportError:
    pass
try:  # pragma: no cover - incremental build
    from .pipeline import WorldFlow3DPipeline  # noqa: F401
    __all__.append("WorldFlow3DPipeline")
except ImportError:
    pass

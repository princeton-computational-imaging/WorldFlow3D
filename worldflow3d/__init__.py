"""WorldFlow3D — hierarchical 3D scene generation, diffusers-native inference.

Standalone, self-contained package (no dependency on the research `src/` tree).
The vendored subpackages ``models``/``conditioning``/``recon`` are generated from
the research repo by ``tools/vendor.py``; ``pipeline``/``convert``/``_remote`` are
hand-authored.
"""
__version__ = "0.1.0.dev0"

from .pipeline.scheduler import WorldFlow3DFlowScheduler  # noqa: F401

__all__ = ["WorldFlow3DFlowScheduler", "__version__"]

# The UNet wrapper and pipeline are added as the build progresses.
try:  # pragma: no cover - incremental build
    from .pipeline.unet import WorldFlow3DUNet  # noqa: F401
    __all__.append("WorldFlow3DUNet")
except ImportError:
    pass
try:  # pragma: no cover - incremental build
    from .pipeline.pipeline import WorldFlow3DPipeline  # noqa: F401
    __all__.append("WorldFlow3DPipeline")
except ImportError:
    pass

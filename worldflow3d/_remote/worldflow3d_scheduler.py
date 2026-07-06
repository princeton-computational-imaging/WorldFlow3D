"""Remote-code mirror for ``WorldFlow3DFlowScheduler`` (diffusers ``trust_remote_code``).

Referenced from ``model_index.json`` as the ``scheduler`` component::

    "scheduler": ["worldflow3d_scheduler", "WorldFlow3DFlowScheduler"]

and placed inside the component subfolder ``<repo>/scheduler/worldflow3d_scheduler.py``
(see ``worldflow3d_unet.py`` for the resolution rule diffusers 0.29.2 uses).

The scheduler is pure-python (numpy + torch, no custom kernels) so it *could* be
made fully self-contained here. We keep it a thin re-export of the installed
``worldflow3d`` class for a single source of truth: that guarantees the Hub load
path and the local ``from_converted`` path use byte-identical step math. If you
ever need a zero-dependency remote scheduler, inline the body of
``worldflow3d.pipeline.scheduler`` here — it has no project-internal imports.
"""
try:
    from worldflow3d.pipeline.scheduler import WorldFlow3DFlowScheduler
except ImportError as exc:  # pragma: no cover - actionable message at load time
    raise ImportError(
        "Loading a WorldFlow3D model via trust_remote_code requires the "
        "'worldflow3d' package to be installed. Install it first, e.g.\n"
        "    pip install --index-url https://test.pypi.org/simple/ worldflow3d   # private dry-run\n"
        "    pip install worldflow3d                                             # public"
    ) from exc

__all__ = ["WorldFlow3DFlowScheduler"]

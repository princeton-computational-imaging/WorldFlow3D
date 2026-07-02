"""Remote-code mirror for ``WorldFlow3DPipeline`` (diffusers ``trust_remote_code``).

This is the top-level custom-pipeline file diffusers downloads + executes when the
repo's ``model_index.json`` declares a *list* ``_class_name``::

    "_class_name": ["worldflow3d_pipeline", "WorldFlow3DPipeline"]

In diffusers 0.29.2 (``pipeline_utils.py:744-751``) a list ``_class_name`` makes the
loader look for ``<repo>/worldflow3d_pipeline.py`` at the **repo root** (not inside a
subfolder), import it via ``get_class_from_dynamic_module``, and use
``_class_name[1]`` as the class. ``trust_remote_code=True`` must be passed.

NOTE: when ``_class_name`` is a plain string (``"WorldFlow3DPipeline"``) the loader
instead requires the class to be importable from the installed ``diffusers`` or the
class you call ``from_pretrained`` on — which is why we publish with the *list* form
so a bare ``DiffusionPipeline.from_pretrained(repo, trust_remote_code=True)`` works
without the caller importing ``worldflow3d`` first.

Like the component mirrors, this re-exports the real class. The pipeline body pulls
in the vendored ``conditioning``/``recon`` subpackages, so the installed
``worldflow3d`` package is required (TestPyPI private dry-run, then PyPI).
"""
try:
    from worldflow3d.pipeline.pipeline import WorldFlow3DPipeline
except ImportError as exc:  # pragma: no cover - actionable message at load time
    raise ImportError(
        "Loading a WorldFlow3D pipeline via trust_remote_code requires the "
        "'worldflow3d' package to be installed (the remote pipeline file re-exports "
        "the real class, which depends on the vendored conditioning/recon "
        "subpackages). Install it first, e.g.\n"
        "    pip install --index-url https://test.pypi.org/simple/ worldflow3d   # private dry-run\n"
        "    pip install worldflow3d                                             # public"
    ) from exc

__all__ = ["WorldFlow3DPipeline"]

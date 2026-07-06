"""Remote-code mirror for ``WorldFlow3DUNet`` (diffusers ``trust_remote_code``).

This single file is what diffusers downloads + executes when loading a published
WorldFlow3D repo from the Hub. It is referenced from ``model_index.json`` as the
``unet`` component::

    "unet": ["worldflow3d_unet", "WorldFlow3DUNet"]

and must be placed **inside the component subfolder** of the repo, i.e.
``<repo>/unet/worldflow3d_unet.py``. diffusers 0.29.2 resolves a component
``[library_name, class_name]`` by first checking
``os.path.isfile(<component_folder>/<library_name>.py)`` — if present it loads the
class via ``get_class_from_dynamic_module`` (the trust-remote-code path); otherwise
it falls back to ``importlib.import_module(library_name)`` (which fails for a
custom top-level module name like ``worldflow3d_unet``). Hence the file lives next
to ``unet/config.json``.

Contract: this is a *thin re-export* of the real class from the installed
``worldflow3d`` package. The full UNet torso lives in ``worldflow3d.models`` (a
multi-file subpackage with sparse-conv / attention kernels) which is impractical to
mirror as a single remote file, so the published-Hub flow REQUIRES the
``worldflow3d`` package to be pip-installed (TestPyPI for the private dry-run, then
PyPI). The re-exported class is a genuine ``ModelMixin``/``ConfigMixin`` subclass,
so ``WorldFlow3DUNet.from_pretrained(<repo>/unet)`` round-trips identically whether
diffusers resolves the class from here or from the installed package.
"""
try:
    from worldflow3d.pipeline.unet import WorldFlow3DUNet, WorldFlow3DUNetOutput
except ImportError as exc:  # pragma: no cover - actionable message at load time
    raise ImportError(
        "Loading a WorldFlow3D model via trust_remote_code requires the "
        "'worldflow3d' package to be installed (the remote files re-export the "
        "real classes from it; the UNet torso cannot be a single remote file). "
        "Install it first, e.g.\n"
        "    pip install --index-url https://test.pypi.org/simple/ worldflow3d   # private dry-run\n"
        "    pip install worldflow3d                                             # public"
    ) from exc

__all__ = ["WorldFlow3DUNet", "WorldFlow3DUNetOutput"]

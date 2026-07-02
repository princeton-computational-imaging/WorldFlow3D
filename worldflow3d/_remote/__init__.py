"""Single-file remote-code mirrors for diffusers ``trust_remote_code`` loading.

These three modules are *not* imported as part of normal package use. They are the
exact files that get copied into a published Hub repo so that
``DiffusionPipeline.from_pretrained(repo_id, trust_remote_code=True)`` can resolve
the custom pipeline + components by downloading and exec'ing remote code:

    <repo>/worldflow3d_pipeline.py        # top-level custom pipeline (_class_name[0])
    <repo>/unet/worldflow3d_unet.py       # custom component "unet"
    <repo>/scheduler/worldflow3d_scheduler.py  # custom component "scheduler"

``worldflow3d.convert.push_to_hub`` copies them to those locations. Each mirror is a
thin re-export of the installed ``worldflow3d`` class (see each module's docstring
for why a fully self-contained mirror is impractical for the UNet/pipeline).

``REMOTE_FILES`` maps source mirror -> destination path *relative to the repo root*
so the push tooling has a single source of truth for placement.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# source filename -> destination path relative to the model repo root
REMOTE_FILES = {
    "worldflow3d_pipeline.py": "worldflow3d_pipeline.py",
    "worldflow3d_unet.py": os.path.join("unet", "worldflow3d_unet.py"),
    "worldflow3d_scheduler.py": os.path.join("scheduler", "worldflow3d_scheduler.py"),
}


def remote_file_source(name: str) -> str:
    """Absolute path to a bundled remote mirror file."""
    return os.path.join(_HERE, name)


__all__ = ["REMOTE_FILES", "remote_file_source"]

"""Publish a converted WorldFlow3D model dir to the Hugging Face Hub.

PRIVATE-FIRST WORKFLOW (do this before anything is public)
----------------------------------------------------------
Both the *model* and the *package* have a private dry-run stage:

  1. Model -> a PRIVATE HF repo. This module pushes with ``private=True`` by default.
     You must pass ``--public`` (and mean it) to publish openly. Validate with
     ``--test``, which reloads the just-pushed repo via
     ``DiffusionPipeline.from_pretrained(repo_id, trust_remote_code=True)`` (NO
     generation — CPU-safe, weights + remote code resolution only).

  2. Package (``worldflow3d``) -> TestPyPI before PyPI. The trust-remote-code load
     path re-exports the real classes from the installed ``worldflow3d`` package, so
     the published model is only loadable once the package is installable. Dry-run:
         python -m build
         twine upload --repository testpypi dist/*
         pip install --index-url https://test.pypi.org/simple/ worldflow3d
     Only after BOTH private dry-runs pass do you flip the model repo to public
     (re-push with ``--public``) and ``twine upload dist/*`` to real PyPI.

WHAT GETS PUSHED
----------------
The converted dir (``model_index.json`` + ``unet/`` + ``scheduler/`` [+ ``vae/``])
PLUS the three single-file remote-code mirrors from ``worldflow3d/_remote/`` copied
to the exact locations diffusers' ``trust_remote_code`` loader expects:

    <repo>/worldflow3d_pipeline.py             (top-level custom pipeline)
    <repo>/unet/worldflow3d_unet.py            (custom "unet" component)
    <repo>/scheduler/worldflow3d_scheduler.py  (custom "scheduler" component)

For the Hub trust-remote-code path to work, ``model_index.json`` must use the LIST
form of ``_class_name`` and LIST component entries (see
``worldflow3d/convert/convert_checkpoint.py`` / the converter spec). This tool will
copy the remote files but will NOT rewrite ``model_index.json`` — emit it correctly
at convert time. ``--check-model-index`` warns if the form looks Hub-incompatible.

USAGE
-----
    # private dry-run + reload test (default)
    python -m worldflow3d.convert.push_to_hub \
        --model-dir outputs/worldflow3d_validation/diff_remote \
        --repo-id <user>/worldflow3d-waymo-coarse \
        --token $HF_TOKEN --test

    # go public (only after TestPyPI + private model dry-run both pass)
    python -m worldflow3d.convert.push_to_hub \
        --model-dir ... --repo-id ... --token $HF_TOKEN --public
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# _remote is a sibling top-level subpackage; import its placement manifest.
try:
    from worldflow3d._remote import REMOTE_FILES, remote_file_source
except ImportError:  # pragma: no cover - allow running as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from worldflow3d._remote import REMOTE_FILES, remote_file_source


def stage_remote_files(model_dir: str, dest_root: str,
                       include_remote_code: bool = True) -> List[str]:
    """Copy the converted model dir (+ optionally the ``_remote`` mirrors) into ``dest_root``.

    Returns the list of repo-relative paths of the remote ``*.py`` mirrors that were
    placed. The converted dir is copied verbatim; if ``include_remote_code`` the remote
    mirrors are overlaid at the locations diffusers' trust-remote-code loader resolves.

    The remote ``*.py`` shims are ONLY needed for the bare-``diffusers``
    ``trust_remote_code`` load path, which requires a model at a **repo root**. For the
    subfolder/cascade layout loaded via the installed package
    (``WorldFlow3DPipeline.from_hub`` → ``from_converted``), they are dead weight — the
    package supplies the classes — so pass ``include_remote_code=False`` to omit them
    (and strip any that leaked in from the converted dir). The model code then lives
    exactly once, in the installed ``worldflow3d`` package.
    """
    model_dir = os.path.abspath(model_dir)
    if not os.path.isfile(os.path.join(model_dir, "model_index.json")):
        raise FileNotFoundError(
            f"{model_dir} is not a converted model dir (no model_index.json). "
            f"Run worldflow3d.convert.convert_checkpoint first."
        )
    # copy the converted layout (model_index.json, unet/, scheduler/, optional vae/)
    for entry in os.listdir(model_dir):
        src = os.path.join(model_dir, entry)
        dst = os.path.join(dest_root, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    placed = []
    if include_remote_code:
        # overlay remote-code mirrors at their canonical repo-relative destinations
        for src_name, rel_dest in REMOTE_FILES.items():
            dst = os.path.join(dest_root, rel_dest)
            os.makedirs(os.path.dirname(dst) or dest_root, exist_ok=True)
            shutil.copy2(remote_file_source(src_name), dst)
            placed.append(rel_dest)
    else:
        # ensure no remote-code shims leaked in from the converted dir
        for rel_dest in REMOTE_FILES.values():
            p = os.path.join(dest_root, rel_dest)
            if os.path.isfile(p):
                os.remove(p)
    return placed


def _check_model_index(repo_root: str) -> Tuple[bool, str]:
    """Sanity-check ``model_index.json`` for Hub trust-remote-code compatibility."""
    import json

    mi = json.load(open(os.path.join(repo_root, "model_index.json")))
    cls = mi.get("_class_name")
    ok = isinstance(cls, (list, tuple)) and len(cls) == 2
    comp_ok = all(
        isinstance(mi.get(k), (list, tuple)) and len(mi[k]) == 2
        for k in ("unet", "scheduler")
    )
    if ok and comp_ok:
        return True, f"_class_name={cls} (list form, Hub-compatible)"
    return False, (
        f"_class_name={cls!r}: for trust_remote_code Hub loading the converter must "
        f'emit a LIST _class_name (e.g. ["worldflow3d_pipeline","WorldFlow3DPipeline"]) '
        f"and list component entries. The installed-class load path still works with a "
        f"string, but DiffusionPipeline.from_pretrained(trust_remote_code=True) will not."
    )


def push_to_hub(
    model_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: str = "Upload WorldFlow3D model (converted + remote code)",
    create: bool = True,
    check_model_index: bool = True,
    subfolder: Optional[str] = None,
    include_remote_code: bool = True,
) -> str:
    """Push a converted model dir (+ remote-code mirrors) to an HF model repo.

    Args:
        model_dir: converter output dir (has ``model_index.json``).
        repo_id: ``<user-or-org>/<name>`` target repo.
        token: HF token (else falls back to ``HF_TOKEN`` env / cached login).
        private: create/keep the repo PRIVATE (default True — private-first).
        commit_message: upload commit message.
        create: ``create_repo(exist_ok=True)`` before uploading.
        check_model_index: warn if ``model_index.json`` is not Hub-compatible.
        subfolder: if given, place this model under ``<repo>/<subfolder>/`` instead
            of the repo root. Lets one repo hold a whole cascade (e.g.
            ``waymo-coarse/``, ``waymo-refine/``, ``front3d-coarse/``); each subfolder
            is a self-contained diffusers layout loadable via
            ``from_pretrained(repo_id, subfolder=...)``.

    Returns:
        The repo URL.
    """
    from huggingface_hub import create_repo, upload_folder

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    with tempfile.TemporaryDirectory(prefix="wf3d_push_") as staging:
        placed = stage_remote_files(model_dir, staging, include_remote_code=include_remote_code)
        ok, msg = _check_model_index(staging)
        if check_model_index and not ok:
            print(f"[push_to_hub][WARN] {msg}")
        else:
            print(f"[push_to_hub] {msg}")
        print(f"[push_to_hub] staged remote files: {placed}")

        if create:
            create_repo(repo_id, token=token, private=private, repo_type="model", exist_ok=True)
        url = upload_folder(
            repo_id=repo_id,
            folder_path=staging,
            path_in_repo=subfolder or "",
            token=token,
            repo_type="model",
            commit_message=commit_message,
        )
    visibility = "PRIVATE" if private else "PUBLIC"
    dest = f"{repo_id}/{subfolder}" if subfolder else repo_id
    print(f"[push_to_hub] pushed {model_dir} -> {dest} ({visibility})")
    return str(url)


def test_load(repo_id: str, token: Optional[str] = None,
              subfolder: Optional[str] = None) -> bool:
    """Reload a pushed repo (NO generation; CPU-safe). Returns True on success.

    Validates the private dry-run end to end. Two paths, picked by layout:

    - **root model** (``subfolder=None``): the bare-``diffusers``
      ``DiffusionPipeline.from_pretrained(repo, trust_remote_code=True)`` one-liner —
      downloads ``model_index.json`` + the remote ``*.py`` mirrors + weights, executes
      the remote code, and instantiates the pipeline (no package install needed).
    - **subfolder stage**: the installed-package path
      ``WorldFlow3DPipeline.from_hub(repo, stage=subfolder)``. diffusers does not apply
      ``subfolder`` at the pipeline level (it reads ``model_index.json`` only at the
      repo root), so the subfolder/cascade layout is loaded via the package.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if subfolder:
        from ..pipeline.pipeline import WorldFlow3DPipeline
        pipe = WorldFlow3DPipeline.from_hub(repo_id, stage=subfolder, device="cpu", token=token)
    else:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(repo_id, trust_remote_code=True, token=token)
    meta = dict(getattr(pipe.unet.config, "level_meta", {}) or {})
    print(
        f"[test_load] OK: {type(pipe).__name__} "
        f"(unet={type(pipe.unet).__name__}, scheduler={type(pipe.scheduler).__name__}, "
        f"level={meta.get('level_name')}, dataset={meta.get('dataset_type')})"
    )
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="converter output dir (has model_index.json)")
    ap.add_argument("--repo-id", required=True, help="<user-or-org>/<name>")
    ap.add_argument("--subfolder", default=None,
                    help="place the model under <repo>/<subfolder>/ (one repo = whole cascade)")
    ap.add_argument("--no-remote-code", action="store_true",
                    help="omit the trust_remote_code *.py shims (package-only load via from_hub; "
                         "recommended for subfolder/cascade pushes — the code lives in the pip package)")
    ap.add_argument("--token", default=None, help="HF token (else $HF_TOKEN / cached login)")
    ap.add_argument("--public", action="store_true",
                    help="publish PUBLICLY (default is PRIVATE; only flip after the private + TestPyPI dry-runs pass)")
    ap.add_argument("--no-create", action="store_true", help="do not create_repo (assume it exists)")
    ap.add_argument("--commit-message", default="Upload WorldFlow3D model (converted + remote code)")
    ap.add_argument("--test", action="store_true",
                    help="after push, reload via DiffusionPipeline.from_pretrained(trust_remote_code=True) — no generation")
    ap.add_argument("--test-only", action="store_true", help="skip push; only run the reload test on --repo-id")
    args = ap.parse_args()

    if not args.test_only:
        url = push_to_hub(
            model_dir=args.model_dir,
            repo_id=args.repo_id,
            token=args.token,
            private=not args.public,
            commit_message=args.commit_message,
            create=not args.no_create,
            subfolder=args.subfolder,
            include_remote_code=not args.no_remote_code,
        )
        print(f"[push_to_hub] repo: {url}")
    if args.test or args.test_only:
        test_load(args.repo_id, token=args.token, subfolder=args.subfolder)


if __name__ == "__main__":
    main()

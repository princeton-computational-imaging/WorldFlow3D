# Releasing `worldflow3d` to PyPI

Publishing uses **Trusted Publishing** (PyPI OIDC) via `.github/workflows/publish.yml`
— no API tokens are stored. A tag publishes to **TestPyPI** (dry-run); a GitHub
**Release** publishes to **real PyPI**.

## One-time setup (per registry)

Register a **publisher** so the workflow is allowed to upload. Since the project
doesn't exist yet, register it as a **"pending publisher"** with exactly these values:

| Field | Value |
|---|---|
| PyPI Project Name | `worldflow3d` |
| Owner | `princeton-computational-imaging` |
| Repository name | `WorldFlow3D` |
| Workflow name | `publish.yml` |
| Environment name | `testpypi` (on TestPyPI) / `pypi` (on PyPI) |

- **TestPyPI:** <https://test.pypi.org/manage/account/publishing/> — environment `testpypi`.
- **PyPI:** <https://pypi.org/manage/account/publishing/> — environment `pypi`.

(TestPyPI and PyPI are separate sites with separate accounts.) Optionally add a
required reviewer to the GitHub `pypi` environment for a manual approval gate.

## Cutting a release

1. Bump `version` in `pyproject.toml`.
2. **Dry-run → TestPyPI:** push a tag.
   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```
   Verify in a clean env:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ worldflow3d
   ```
3. **Real PyPI:** create a GitHub Release from that tag (the `release: published`
   trigger runs the `pypi` job).

The build ships only the `worldflow3d/` package (wheel + sdist), validated by
`twine check`.

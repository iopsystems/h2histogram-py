# Releasing

`h2histogram` is published to [PyPI](https://pypi.org/project/h2histogram/)
automatically by the [`release.yml`](.github/workflows/release.yml) GitHub
Actions workflow, using **PyPI Trusted Publishing (OIDC)** — there are no API
tokens or repository secrets to manage.

## Cutting a release

1. **Bump the version.** Edit `version` in [`pyproject.toml`](pyproject.toml)
   following [semantic versioning](https://semver.org/) (e.g. `0.1.0` → `0.1.1`
   for fixes, `0.2.0` for backwards-compatible features). Commit and merge to
   `main` via a pull request.
2. **Publish a GitHub Release.** Go to
   [Releases → Draft a new release](https://github.com/iopsystems/h2histogram-py/releases/new):
   - **Choose a tag:** type `vX.Y.Z` (matching the `pyproject.toml` version) and
     select *"Create new tag on publish"*, targeting `main`.
   - **Title:** `vX.Y.Z`.
   - Click **Generate release notes**, then **Publish release**.
3. **Watch the workflow.** Publishing the release triggers `release.yml`, which:
   - builds the sdist and wheel,
   - runs `twine check` on them,
   - publishes to PyPI from the `publish` job (which runs in the `pypi`
     environment) via OIDC.

   Follow it under the
   [Actions tab](https://github.com/iopsystems/h2histogram-py/actions/workflows/release.yml).
   Within a minute or so, `pip install h2histogram==X.Y.Z` should work.

> **Versions are immutable on PyPI.** A given version can never be re-uploaded,
> only [yanked](https://pypi.org/help/#yanked). If you push a bad release, bump
> the version and cut a new one. Do a dry run on
> [TestPyPI](https://test.pypi.org/) first if you're unsure.

## One-time setup (already done, for reference)

Trusted publishing was configured on PyPI under **Account → Publishing → Add a
pending publisher** with:

| Field            | Value              |
|------------------|--------------------|
| Owner            | `iopsystems`       |
| Repository name  | `h2histogram-py`   |
| Workflow name    | `release.yml`      |
| Environment name | `pypi`             |

The first successful release created the project on PyPI and converted the
pending publisher into a live one.

## Moving the project to the PyPI organization

The project is currently owned by the account that first published it. To move
it under the `iopsystems` PyPI organization (once that organization is approved):

- Project → **Manage → Settings → Transfer project to an organization** →
  select `iopsystems`.

The trusted-publisher binding survives the transfer, so releases keep working
unchanged.

## Verifying a release locally

```bash
python -m venv /tmp/verify && /tmp/verify/bin/pip install "h2histogram[parquet]==X.Y.Z"
/tmp/verify/bin/python -c "import h2histogram; print(h2histogram.__version__)"
```

# Releasing `mitrity`

A release is a tag `vX.Y.Z` on a commit of `main`. Pushing it runs
[`.github/workflows/release.yml`](.github/workflows/release.yml), which builds the
sdist and wheel, publishes them to PyPI through
[trusted publishing](https://docs.pypi.org/trusted-publishers/) (PyPI verifies the
workflow's OIDC token; no API token exists anywhere), and creates the GitHub
Release with the version's CHANGELOG section as notes and the distributions
attached.

## One-time setup (founder)

1. **PyPI account**: two-factor authentication enabled on the account that will
   own the project.
2. **Pending publisher**: <https://pypi.org/manage/account/publishing/>, form
   "GitHub":

   | Field | Value |
   | --- | --- |
   | PyPI project name | `mitrity` |
   | Owner | `mitrity-io` |
   | Repository name | `mitrity-python` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   A pending publisher does not reserve the name: the project `mitrity` is
   created by the first publish, and the pending publisher becomes the project's
   trusted publisher at that moment. The name was free on 2026-09-19; check
   <https://pypi.org/project/mitrity/> still returns 404 before the first tag.
3. **GitHub environment `pypi`**: repository Settings → Environments → New
   environment → `pypi`. Under "Deployment branches and tags" choose "Selected
   branches and tags" and add the tag rule `v*`, so no branch can ever use the
   environment. Required reviewers are available on public repositories (and on
   private ones only with GitHub Enterprise): once the repository is public, add
   yourself as a required reviewer so every publish waits for one approval click.
4. **Tag ruleset (recommended)**: Settings → Rules → Rulesets → New tag ruleset
   for `v*` that restricts creation, update and deletion to repository admins.
   Whoever can push a matching tag can publish.

## Each release

1. Branch from `origin/main`:
   `git fetch origin && git switch -c release/vX.Y.Z origin/main`.
2. Set the version in `src/mitrity/__init__.py` (`__version__ = "X.Y.Z"`).
   `pyproject.toml` reads it from there through hatch; nothing else carries it.
3. `CHANGELOG.md`: move the `[Unreleased]` items under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading with today's date, leave `## [Unreleased]`
   empty above it, and add the two link references at the bottom. For the first
   release, replace `unreleased` in the `0.1.0` heading with the date.
4. Pre-flight locally, exactly what the workflow will check:

   ```bash
   TAG=vX.Y.Z python .github/scripts/release_check.py
   python -m pip install "build==1.6.1" "twine==7.0.0"
   rm -rf dist && python -m build && python -m twine check --strict dist/*
   ```

5. Open the PR (`chore(release): vX.Y.Z`), let the review pipeline approve it,
   merge.
6. Tag the merge commit on `main`:

   ```bash
   git fetch origin && git switch main && git pull --ff-only
   gh api repos/mitrity-io/mitrity-python/git/ref/tags/vX.Y.Z   # must fail: a tag is never moved
   git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
   ```

7. Watch the run under
   <https://github.com/mitrity-io/mitrity-python/actions/workflows/release.yml>;
   approve the `pypi` environment when a reviewer is configured.
8. Verify: <https://pypi.org/project/mitrity/X.Y.Z/> lists the sdist and wheel
   with attestations ("Verified details" names this repository and workflow),
   `pip install mitrity==X.Y.Z` works in a clean virtual environment, and the
   GitHub Release carries both files.

## What the workflow refuses

- A tag whose commit is not on `main`.
- A tag that is not `v` + the package version, or a version without a dated,
  non-empty `CHANGELOG.md` section.
- Anything in `dist/` other than the tagged version's sdist and wheel, or
  distributions that fail `twine check --strict`.

## Pre-releases

Tag `v0.2.0rc1` with `__version__ = "0.2.0rc1"` (the PEP 440 normalized form,
which is also the changelog heading). The GitHub Release is marked as a
pre-release; PyPI hides it from `pip install mitrity` until `--pre` is passed.

## When a run fails

- `build` failed, or `publish` failed before uploading (for example PyPI
  answered `invalid-publisher`: a field of the pending publisher does not match
  the table above): nothing left the repository. Fix the cause and use "Re-run
  failed jobs"; if the fix needs a commit, delete the tag
  (`git push --delete origin vX.Y.Z`) and tag again once it is on `main`. This is
  the only situation in which a tag is deleted.
- `publish` succeeded and `release` failed: "Re-run failed jobs". The run's
  artifacts are reused; PyPI is not touched.
- The version is on PyPI and is broken: files on PyPI are immutable, so yank
  the release there (Manage → Releases → Options → Yank), fix forward and
  release the next patch version. Never move a tag that has been published.

## Tooling versions

The workflow pins `build` and `twine` to exact versions and the actions to
commit SHAs. Dependabot bumps the actions; bump the two Python tools
deliberately in `release.yml` and in the pre-flight command above.

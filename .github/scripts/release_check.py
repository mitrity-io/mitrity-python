"""Release pre-flight: the tag must name the package version and have a changelog entry.

.github/workflows/release.yml runs this before anything is installed or built (on Python
3.13; the script needs 3.11 or newer for ``tomllib``). Run it by hand before tagging:

    TAG=v0.1.0 python .github/scripts/release_check.py

It checks, in order, that

- the version pyproject.toml resolves to (a static ``project.version``, or the
  ``__version__`` in the file hatch's ``tool.hatch.version.path`` names) equals the
  tag without its leading ``v``;
- CHANGELOG.md has a dated ``## [<version>] - YYYY-MM-DD`` section with content.

``--notes-file PATH`` writes that section to PATH; the workflow uses it as the notes
of the GitHub Release. When ``GITHUB_OUTPUT`` is set, ``version=`` and ``prerelease=``
are appended to it. Any failure exits with status 1 and the reason on stderr.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import NoReturn

import tomllib  # Python 3.11+, which ruff's py310 target files outside the standard library

ROOT = Path(__file__).resolve().parents[2]

# A final release is X.Y.Z, optionally with a PEP 440 post-release suffix. Anything
# else (0.2.0rc1, 0.2.0a1, 0.2.0.dev3) is marked as a pre-release on GitHub.
FINAL_VERSION = re.compile(r"^\d+\.\d+\.\d+(\.post\d+)?$")
VERSION_ASSIGNMENT = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)
LINK_DEFINITION = re.compile(r"^\[[^\]]+\]:\s*\S+")


def fail(reason: str) -> NoReturn:
    raise SystemExit(f"release check: {reason}")


def project_version(pyproject: Path) -> str:
    """The version pyproject.toml declares, statically or through hatch's version file."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    if isinstance(version, str):
        return version
    if "version" not in project.get("dynamic", []):
        fail("pyproject.toml declares no project version")
    path = data.get("tool", {}).get("hatch", {}).get("version", {}).get("path")
    if not isinstance(path, str):
        fail("pyproject.toml has a dynamic version but no [tool.hatch.version] path")
    match = VERSION_ASSIGNMENT.search((pyproject.parent / path).read_text(encoding="utf-8"))
    if match is None:
        fail(f"{path} has no __version__ assignment")
    return match.group(1)


def changelog_section(changelog: Path, version: str) -> str:
    """The body of CHANGELOG.md's dated section for ``version``."""
    heading = re.compile(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$")
    body: list[str] = []
    found = False
    for line in changelog.read_text(encoding="utf-8").splitlines():
        if not found:
            found = heading.match(line) is not None
            continue
        if line.startswith("## ") or LINK_DEFINITION.match(line):
            break
        body.append(line)
    if not found:
        fail(
            f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section; "
            "add the entry, with the release date, before tagging"
        )
    text = "\n".join(body).strip()
    if not text:
        fail(f"the CHANGELOG.md section for {version} is empty")
    return text + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check that a release tag matches the package version and changelog."
    )
    parser.add_argument(
        "--tag",
        default=os.environ.get("TAG") or os.environ.get("GITHUB_REF_NAME"),
        help="the release tag, e.g. v0.1.0 (default: $TAG, then $GITHUB_REF_NAME)",
    )
    parser.add_argument(
        "--notes-file", type=Path, help="write the changelog section of the version here"
    )
    args = parser.parse_args(argv)
    if not args.tag:
        parser.error("no tag: pass --tag vX.Y.Z or set TAG")

    version = project_version(ROOT / "pyproject.toml")
    if args.tag != f"v{version}":
        fail(f"tag {args.tag} does not match the package version {version} (expected v{version})")
    notes = changelog_section(ROOT / "CHANGELOG.md", version)
    prerelease = FINAL_VERSION.match(version) is None

    if args.notes_file is not None:
        args.notes_file.write_text(notes, encoding="utf-8")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"version={version}\nprerelease={str(prerelease).lower()}\n")

    kind = "pre-release" if prerelease else "final release"
    print(f"tag {args.tag} matches version {version} ({kind})")
    print(f"CHANGELOG.md section for {version}: {len(notes.splitlines())} lines")


if __name__ == "__main__":
    main()

"""Small, dependency-free provenance helpers shared by benchmark scripts.

Version strings alone do not identify an editable checkout.  In particular,
the distribution metadata can lag behind the source currently imported.  Each
new benchmark run therefore records the repository commit, the producer
script, and one digest over every benchmark and package source that can affect
the result.  The tree digest still identifies an intentional pre-commit run
when ``source_dirty`` is true.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _sha256(path, chunk=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _source_digest(root, files):
    """Stable SHA-256 over relative names and contents of source files."""
    digest = hashlib.sha256()
    for path in sorted({Path(path).resolve() for path in files}):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def benchmark_identity(script):
    """Return the source identity that should accompany one benchmark run.

    ``source_dirty`` is scoped to benchmark Python sources, the package code,
    and its build metadata.  Result files are deliberately excluded: a
    benchmark normally rewrites those before constructing its provenance.
    """
    script = Path(script).resolve()
    root = script.parents[1]
    try:
        relative = script.relative_to(root).as_posix()
    except ValueError:
        relative = script.name
    benchmark_files = list(Path(__file__).resolve().parent.glob("*.py"))
    package_files = list((root / "multipgs").rglob("*.py"))
    source_files = [*benchmark_files, *package_files, root / "pyproject.toml"]
    benchmark_sources = sorted(
        path.relative_to(root).as_posix() for path in benchmark_files)
    paths = [*benchmark_sources, "multipgs", "pyproject.toml"]
    status = _git(root, "status", "--porcelain", "--untracked-files=normal",
                  "--", *paths)
    return {
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "source_dirty": None if status is None else bool(status),
        "dirty_scope": paths,
        "source_sha256": _source_digest(root, source_files),
        "script": relative,
        "script_sha256": _sha256(script),
    }

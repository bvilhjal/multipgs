"""The documentation must not drift away from the code or the bibliography."""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md",
                                               ROOT / "CHANGELOG.md"]
DOI_RE = re.compile(r"https://doi\.org/(10\.[^\s)\]>`]+)")


def _dois(text):
    return {m.group(1).rstrip(".,;") for m in DOI_RE.finditer(text)}


def test_every_doi_cited_in_the_docs_is_in_the_bibliography():
    """A citation that is not in references.md was never verified."""
    bib = _dois((ROOT / "docs" / "references.md").read_text(encoding="utf-8"))
    assert len(bib) > 50, "references.md looks truncated"
    for path in DOCS:
        if path.name == "references.md":
            continue
        missing = _dois(path.read_text(encoding="utf-8")) - bib
        assert not missing, (f"{path.name} cites DOIs absent from "
                             f"references.md: {sorted(missing)}")


def test_source_docstrings_cite_only_verified_dois():
    bib = _dois((ROOT / "docs" / "references.md").read_text(encoding="utf-8"))
    for path in sorted((ROOT / "multipgs").glob("*.py")):
        missing = _dois(path.read_text(encoding="utf-8")) - bib
        assert not missing, (f"{path.name} cites DOIs absent from "
                             f"references.md: {sorted(missing)}")


def test_bibliography_entries_are_well_formed():
    text = (ROOT / "docs" / "references.md").read_text(encoding="utf-8")
    entries = [ln for ln in text.splitlines() if ln.startswith("- **")]
    assert len(entries) >= 80
    for line in entries:
        assert re.search(r"\*\*.+\*\* \(\d{4}\)\.", line), line
        assert "*" in line.split(").", 1)[1], f"no venue: {line}"


@pytest.mark.parametrize("name", ["theory.md", "guide.md", "algorithm.md",
                                  "api.md", "references.md"])
def test_docs_exist_and_are_linked_from_the_readme(name):
    assert (ROOT / "docs" / name).exists()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"docs/{name}" in readme, f"README does not link docs/{name}"


def test_internal_doc_links_resolve():
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", text):
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"{path.name} links to missing {target}"

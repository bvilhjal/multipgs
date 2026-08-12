"""The documentation must not drift away from the code or the bibliography."""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md",
                                               ROOT / "CHANGELOG.md"]
DOI_RE = re.compile(r"https://doi\.org/(10\.[^\s)\]>`]+)")


def _dois(text):
    return {m.group(1).rstrip(".,;}") for m in DOI_RE.finditer(text)}


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


def test_technical_review_is_complete_and_linked():
    """The versioned review is a real deliverable, not a dangling link."""
    report = ROOT / "report"
    source = report / "multipgs_review.tex"
    pdf = report / "multipgs_review.pdf"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert source.exists()
    assert pdf.exists()
    assert "report/multipgs_review.tex" in readme
    assert "report/multipgs_review.pdf" in readme
    assert pdf.stat().st_size > 100_000
    assert pdf.read_bytes().startswith(b"%PDF-")

    latex = source.read_text(encoding="utf-8")
    bibliography = _dois(
        (ROOT / "docs" / "references.md").read_text(encoding="utf-8"))
    assert not (_dois(latex) - bibliography)
    assert latex.count(r"\begin{figure}") >= 5
    assert latex.count(r"\begin{table}") + latex.count(r"\begin{longtable}") >= 5
    assert latex.count(r"\begin{equation}") >= 10


def test_internal_doc_links_resolve():
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", text):
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"{path.name} links to missing {target}"


def test_every_benchmark_named_in_the_docs_exists():
    """A documented benchmark that is not in the tree is worse than none.

    ``test_internal_doc_links_resolve`` only inspects markdown links, but a
    benchmark is normally named in prose as ``benchmark.py`` or inside a fenced
    shell command, neither of which is a link. A script deleted or renamed
    without touching its documentation therefore leaves instructions that
    silently fail for whoever follows them.

    Only ``benchmarks/<name>.py`` in *this* repository counts. A sibling
    project's script is referenced with its own prefix — ldpred3's
    ``ldpred3/benchmarks/convert_bigsnpr_ldref.py`` builds the LD reference
    several of these benchmarks take as input — and cannot be checked here.
    """
    scripts = {path.name for path in (ROOT / "benchmarks").glob("*.py")}
    # Not preceded by another path component, which is what distinguishes this
    # repository's benchmarks/ from a sibling project's.
    referenced = re.compile(r"(?<![\w/])benchmarks/([a-z0-9_]+\.py)")
    # benchmarks/README.md is where benchmarks are described, and it is
    # deliberately not in DOCS, which covers the prose documentation set.
    for path in [*DOCS, ROOT / "benchmarks" / "README.md"]:
        text = path.read_text(encoding="utf-8")
        missing = set(referenced.findall(text)) - scripts
        assert not missing, (f"{path.name} gives instructions for benchmark "
                             f"script(s) that do not exist: {sorted(missing)}")


def test_every_benchmark_is_documented():
    """A benchmark nobody can find is evidence nobody will run."""
    readme = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    for script in sorted((ROOT / "benchmarks").glob("*.py")):
        if script.name.startswith("_"):
            continue
        assert script.name in readme, (
            f"benchmarks/{script.name} is not mentioned in "
            "benchmarks/README.md")

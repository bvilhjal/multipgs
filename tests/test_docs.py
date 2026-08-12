"""The documentation must not drift away from the code or the bibliography."""

import importlib.util
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


def test_methods_report_is_complete_and_linked():
    """The packaged methods report is a real deliverable, not an audit."""
    report = ROOT / "report"
    source = report / "multipgs_methods.tex"
    pdf = report / "multipgs_methods.pdf"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert source.exists()
    assert pdf.exists()
    assert "report/multipgs_methods.tex" in readme
    assert "report/multipgs_methods.pdf" in readme
    assert pdf.stat().st_size > 100_000
    assert pdf.read_bytes().startswith(b"%PDF-")

    latex = source.read_text(encoding="utf-8")
    assert "METHODS AND VALIDATION" in latex
    assert r"\section{Implemented methods}" in latex
    assert r"\section{Validation results}" in latex
    assert "Technical review findings" not in latex
    assert "Recommended roadmap" not in latex
    bibliography = _dois(
        (ROOT / "docs" / "references.md").read_text(encoding="utf-8"))
    assert not (_dois(latex) - bibliography)
    assert latex.count(r"\begin{figure}") >= 5
    assert latex.count(r"\begin{table}") + latex.count(r"\begin{longtable}") >= 5
    assert latex.count(r"\begin{equation}") >= 10


def test_report_evidence_is_current_without_rerunning_the_suite():
    generator = ROOT / "report" / "generate_evidence.py"
    spec = importlib.util.spec_from_file_location("report_evidence", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = module.validate_committed_evidence(ROOT)
    assert evidence["tests"]["status"] == "passed"


def test_report_input_order_does_not_depend_on_the_operating_system():
    """The digest's input order must be the same on Windows and POSIX.

    ``sorted()`` over ``Path`` objects compares by flavour: case-folded on
    Windows, case-sensitive on POSIX. Since :func:`report_input_digest` hashes
    the inputs in this order, a flavour-dependent sort would make the recorded
    digest — the report's stated reproducibility authority — disagree with
    itself across platforms, and the committed evidence would fail its own
    freshness check on the other one.
    """
    generator = ROOT / "report" / "generate_evidence.py"
    spec = importlib.util.spec_from_file_location("report_evidence", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    inputs = [path.as_posix() for path in module.report_inputs(ROOT)]
    assert inputs == sorted(inputs)

    # Same relative paths under the other flavour: the order must not move.
    windows = sorted((pathlib.PureWindowsPath(name) for name in inputs),
                     key=pathlib.PureWindowsPath.as_posix)
    assert [path.as_posix() for path in windows] == inputs


def test_internal_doc_links_resolve():
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", text):
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"{path.name} links to missing {target}"


def test_internal_doc_anchors_resolve():
    """Check local Markdown fragments, including duplicate heading suffixes."""
    def anchors(path):
        found = set()
        counts = {}
        for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$",
                                  path.read_text(encoding="utf-8"), re.M):
            slug = re.sub(r"[^\w\- ]", "", heading.lower())
            slug = re.sub(r"\s+", "-", slug.strip())
            number = counts.get(slug, 0)
            counts[slug] = number + 1
            found.add(slug if number == 0 else f"{slug}-{number}")
        return found

    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for target, fragment in re.findall(
                r"\]\((?!https?:|mailto:)([^)#]*)#([^)]+)\)", text):
            destination = (path.parent / target).resolve() if target else path
            assert destination.exists(), f"{path.name} links to missing {target}"
            assert fragment in anchors(destination), (
                f"{path.name} links to missing #{fragment} in "
                f"{destination.name}")


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

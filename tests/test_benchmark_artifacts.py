"""Regression checks for the committed benchmark evidence."""

import importlib.util
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_committed_benchmark_artifacts_are_internally_consistent():
    path = ROOT / "benchmarks" / "check_results.py"
    spec = importlib.util.spec_from_file_location("check_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_results()


def test_every_benchmark_records_source_identity_for_future_runs():
    for path in sorted((ROOT / "benchmarks").glob("*.py")):
        if path.name.startswith("_") or path.name == "check_results.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert '"source": benchmark_identity(__file__)' in text, path.name


def test_benchmark_manifest_is_an_allowlist_not_a_broad_graft():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "graft benchmarks" not in manifest
    for line in ("include benchmarks/README.md",
                 "include benchmarks/*.py",
                 "recursive-include benchmarks/results *.csv *.json"):
        assert line in manifest


def _load_stack_scaling(name):
    path = ROOT / "benchmarks" / "stack_scaling.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_peak_rss_is_available_on_this_platform():
    assert _load_stack_scaling("stack_scaling_portability")._rss_mb() > 0.0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PSAPI contract")
def test_windows_peak_rss_backend_is_positive():
    assert _load_stack_scaling("stack_scaling_windows")._windows_peak_rss_mb() > 0.0


def test_distribution_policy_ships_contributor_and_user_docs():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "include CONTRIBUTING.md" in manifest
    assert "include .gitignore" in manifest
    assert "include .github/workflows/ci.yml" in manifest
    assert "recursive-include docs *.md" in manifest
    assert "recursive-include report *.md *.tex *.pdf *.sha256 *.json *.py" in manifest
    assert "recursive-include tests *.py" in manifest
    for path in ("CONTRIBUTING.md", "docs/guide.md", "docs/api.md",
                 "report/evidence.json", "report/generate_evidence.py",
                 "report/generated_evidence.tex"):
        assert f'"{path}"' in pyproject
    assert "Documentation =" in pyproject


def test_ci_smokes_installed_wheel_outside_checkout():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8")
    assert 'smoke_dir="$(mktemp -d)"' in workflow
    assert "sysconfig.get_paths()['purelib']" in workflow
    assert 'python "$checkout/examples/minimal.py"' in workflow
    assert '"tests/test_api.py"' in workflow
    assert 'glob.glob("tests/*.py")' in workflow


def test_ignore_policy_has_no_obsolete_figure_pipeline():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "make_paper_figures.py" not in ignore
    assert "benchmarks/figures.pdf" not in ignore

"""Regression checks for the committed benchmark evidence."""

import importlib.util
import pathlib


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

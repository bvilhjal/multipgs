#!/usr/bin/env python3
"""Generate and validate reproducible evidence metadata for the methods report.

The report records two complementary provenance identifiers:

* ``source_revision`` is the human-readable Git revision at generation time;
* ``input_digest`` is the reproducibility authority: a SHA-256 over the
  explicitly declared report inputs, with paths and line endings normalized.

Generated files and the PDF are deliberately excluded, avoiding a
self-referential commit hash or digest. Ordinary freshness checks never run the
test suite; regeneration does, unless ``--test-count`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
EVIDENCE_PATH = REPORT / "evidence.json"
GENERATED_TEX_PATH = REPORT / "generated_evidence.tex"
SCHEMA_VERSION = 1

_INPUT_GLOBS = (
    "multipgs/*.py",
    "docs/*.md",
    "tests/*.py",
    "benchmarks/*.py",
    "benchmarks/results/*.csv",
    "benchmarks/results/*_provenance.json",
)
_INPUT_FILES = (
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "MANIFEST.in",
    "report/README.md",
    "report/multipgs_methods.tex",
    "report/generate_evidence.py",
)


def report_inputs(root=ROOT):
    """Return the sorted, declared report inputs relative to ``root``."""
    root = Path(root)
    paths = {root / relative for relative in _INPUT_FILES}
    for pattern in _INPUT_GLOBS:
        paths.update(root.glob(pattern))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = ", ".join(str(path.relative_to(root)) for path in missing)
        raise FileNotFoundError(f"missing declared report input(s): {names}")
    # as_posix() keeps the order independent of pathlib/filesystem case-folding.
    return tuple(sorted((path.relative_to(root) for path in paths),
                        key=lambda path: path.as_posix()))


def report_input_digest(root=ROOT, inputs=None):
    """SHA-256 of normalized paths and contents for the declared inputs."""
    root = Path(root)
    inputs = report_inputs(root) if inputs is None else tuple(map(Path, inputs))
    digest = hashlib.sha256()
    for relative in inputs:
        data = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def package_version(root=ROOT):
    """Read the single-sourced package version without importing the package."""
    tree = ast.parse((Path(root) / "multipgs" / "__init__.py").read_text(
        encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == "__version__" for target in node.targets)):
            return ast.literal_eval(node.value)
    raise RuntimeError("multipgs.__version__ was not found")


def _rows(name, root=ROOT):
    path = Path(root) / "benchmarks" / "results" / name
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row, name):
    return float(row[name])


def _fmt(value, digits):
    return f"{float(value):.{digits}f}"


def _coordinates(rows, x, y, *, digits, x_transform=str):
    return "".join(
        f"({x_transform(row[x])},{_fmt(row[y], digits)})" for row in rows)


def _compact_number(value):
    """Format numeric plot coordinates as TeX writes them (0, 0.1, ...)."""
    return f"{float(value):g}"


def benchmark_evidence(root=ROOT):
    """Derive every checked report headline and plotted coordinate series."""
    fit_raw = _rows("fit_accuracy.csv", root)
    fit = _rows("fit_accuracy_summary.csv", root)
    null = _rows("null_gate_summary.csv", root)
    agreement, = _rows("sumstat_vs_individual_summary.csv", root)
    calibration, = _rows("sumstat_calibration_summary.csv", root)
    real_ld_rows = _rows("real_ld_simulation_summary.csv", root)
    overlap = _rows("overlap_inflation_summary.csv", root)
    meta = _rows("meta_rules_summary.csv", root)
    real_meta, = _rows("real_meta_rules_summary.csv", root)

    fit_wins = sum(_number(row, "multi_r2") > _number(row, "best_single_r2")
                   for row in fit_raw)
    uplift = [_number(row, "uplift_mean") for row in fit]
    gap_closed = [
        (_number(row, "multi_r2_mean") - _number(row, "best_single_r2_mean"))
        / (_number(row, "oracle_r2_mean") - _number(row, "best_single_r2_mean"))
        for row in fit
    ]
    cv_minus_holdout = sum(
        _number(row, "cv_r2") - _number(row, "multi_r2") for row in fit_raw
    ) / len(fit_raw)
    null_total = sum(int(row["n_seeds"]) for row in null)
    null_passes = sum(round((1.0 - _number(row, "null_model_returned_mean"))
                            * int(row["n_seeds"])) for row in null)

    compared_r2 = [
        _number(agreement, key) for key in (
            "individual_holdout_r2_mean",
            "sumstat_holdout_r2_independent_mean",
            "sumstat_holdout_r2_pumas_mean",
            "sumstat_holdout_r2_none_mean",
        )
    ]
    real_ld = {row["reference_n"]: row for row in real_ld_rows}
    ld_small, ld_full = real_ld["50"], real_ld["inf"]
    ld_true_drop = (_number(ld_full, "true_r2_multi_mean")
                    - _number(ld_small, "true_r2_multi_mean"))
    ld_inflation = _number(ld_small, "own_reference_minus_true_mean")
    ld_inflation_pct = 100.0 * ld_inflation / _number(
        ld_small, "true_r2_multi_mean")

    overlap_by_value = {float(row["overlap"]): row for row in overlap}
    overlap_nonnull = [row for row in overlap if float(row["overlap"]) > 0.0]
    detected_nonnull = sum(round(_number(row, "ldsc_detected_mean")
                                 * int(row["n_seeds"]))
                           for row in overlap_nonnull)
    detected_null = round(_number(overlap_by_value[0.0], "ldsc_detected_mean")
                          * int(overlap_by_value[0.0]["n_seeds"]))
    overlap_null_total = int(overlap_by_value[0.0]["n_seeds"])

    def ratio(overlap_value, reported, true):
        row = overlap_by_value[overlap_value]
        return _number(row, reported) / _number(row, true)

    headlines = {
        "fit_replicates": str(len(fit_raw)),
        "fit_wins": str(fit_wins),
        "fit_uplift_min": _fmt(min(uplift), 4),
        "fit_uplift_max": _fmt(max(uplift), 4),
        "fit_gap_closed_min_pct": _fmt(100.0 * min(gap_closed), 1),
        "fit_gap_closed_max_pct": _fmt(100.0 * max(gap_closed), 1),
        "fit_cv_minus_holdout": _fmt(cv_minus_holdout, 5),
        "null_passes": str(null_passes),
        "null_replicates": str(null_total),
        "sumstat_beta_correlation": _fmt(
            agreement["agreement_beta_std_corr_mean"], 6),
        "sumstat_r2_min": _fmt(min(compared_r2), 4),
        "sumstat_r2_max": _fmt(max(compared_r2), 4),
        "pumas_gaussian_error_pct": _fmt(
            100.0 * _number(calibration, "gaussian_plugin_cov_rel_error_mean"),
            2),
        "pumas_binary_error_pct": _fmt(
            100.0 * _number(calibration, "binary_plugin_cov_rel_error_mean"),
            2),
        "ld_true_drop_n50": _fmt(ld_true_drop, 5),
        "ld_report_inflation_n50": _fmt(ld_inflation, 5),
        "ld_report_inflation_n50_pct": _fmt(ld_inflation_pct, 1),
        "overlap_multi_ratio_010": _fmt(ratio(
            0.1, "reported_r2_multi_mean", "true_r2_multi_mean"), 2),
        "overlap_single_ratio_010": _fmt(ratio(
            0.1, "reported_r2_single_mean", "true_r2_single_mean"), 2),
        "overlap_multi_ratio_025": _fmt(ratio(
            0.25, "reported_r2_multi_mean", "true_r2_multi_mean"), 2),
        "overlap_single_ratio_025": _fmt(ratio(
            0.25, "reported_r2_single_mean", "true_r2_single_mean"), 2),
        "overlap_detected_nonnull": str(detected_nonnull),
        "overlap_nonnull_replicates": str(sum(
            int(row["n_seeds"]) for row in overlap_nonnull)),
        "overlap_detected_null": str(detected_null),
        "overlap_null_replicates": str(overlap_null_total),
        "real_meta_expected_r2": _fmt(
            real_meta["r2_expected_r2"], 5),
        "real_meta_decorrelated_expected_r2": _fmt(
            real_meta["r2_decorrelated_expected_r2"], 6),
    }

    fit_labels = iter("ABCDEFGH")
    fit_plot = [{**row, "label": next(fit_labels)} for row in fit]
    ld_order = [real_ld[key] for key in ("50", "200", "1000", "5000", "inf")]
    ld_x = lambda value: "infinite" if value == "inf" else value
    plots = {
        "fit_best_single": _coordinates(
            fit_plot, "label", "best_single_r2_mean", digits=5),
        "fit_multi": _coordinates(
            fit_plot, "label", "multi_r2_mean", digits=5),
        "fit_oracle": _coordinates(
            fit_plot, "label", "oracle_r2_mean", digits=5),
        "ld_true_fitted": _coordinates(
            ld_order, "reference_n", "true_r2_multi_mean", digits=5,
            x_transform=ld_x),
        "ld_own_reference": _coordinates(
            ld_order, "reference_n", "regime_a_r2_own_reference_mean",
            digits=5, x_transform=ld_x),
        "ld_plugin": _coordinates(
            ld_order, "reference_n", "plugin_r2_mean", digits=5,
            x_transform=ld_x),
        "ld_best_single": _coordinates(
            ld_order, "reference_n", "true_r2_best_single_mean", digits=5,
            x_transform=ld_x),
        "overlap_multi_true": _coordinates(
            overlap, "overlap", "true_r2_multi_mean", digits=6,
            x_transform=_compact_number),
        "overlap_multi_reported": _coordinates(
            overlap, "overlap", "reported_r2_multi_mean", digits=6,
            x_transform=_compact_number),
        "overlap_single_true": _coordinates(
            overlap, "overlap", "true_r2_single_mean", digits=6,
            x_transform=_compact_number),
        "overlap_single_reported": _coordinates(
            overlap, "overlap", "reported_r2_single_mean", digits=6,
            x_transform=_compact_number),
        "meta_best_single": _coordinates(
            meta, "shared_error_correlation", "best_single_mean", digits=5,
            x_transform=_compact_number),
        "meta_sqrt_n_eff": _coordinates(
            meta, "shared_error_correlation", "sqrt_n_eff_mean", digits=5,
            x_transform=_compact_number),
        "meta_expected_r2": _coordinates(
            meta, "shared_error_correlation", "expected_r2_mean", digits=5,
            x_transform=_compact_number),
        "meta_decorrelated_true": _coordinates(
            meta, "shared_error_correlation", "decorrelated_expected_r2_mean",
            digits=5, x_transform=_compact_number),
        "real_meta_direct": (
            f"(sqrtN,{_fmt(real_meta['r2_sqrt_n_eff'], 5)})"
            f"(expected,{_fmt(real_meta['r2_expected_r2'], 5)})"),
        "real_meta_decorrelated": (
            f"(sqrtN,{_fmt(real_meta['r2_decorrelated_n_eff'], 6)})"
            f"(expected,{_fmt(real_meta['r2_decorrelated_expected_r2'], 6)})"),
    }
    return {"headlines": headlines, "plots": plots}


def _distribution_version(distribution, module=None):
    if module is not None:
        # Editable-install metadata can lag behind the source actually imported.
        # The report describes the runtime used by the tests, so prefer that
        # module's explicit version when it exposes one.
        imported = __import__(module)
        version = getattr(imported, "__version__", None)
    else:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(
                f"cannot determine {distribution} version") from None
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"cannot determine {distribution} version")
    return version


def test_environment(test_count):
    environment = {
        "status": "passed",
        "count": int(test_count),
        "python": platform.python_version(),
        "numpy": _distribution_version("numpy"),
        "scikit_learn": _distribution_version("scikit-learn", "sklearn"),
        "pytest": _distribution_version("pytest"),
        "ldpred3": _distribution_version("ldpred3", "ldpred3"),
    }
    if any(not isinstance(value, str) or not value.strip()
           for key, value in environment.items()
           if key not in {"count"}):
        raise RuntimeError("test environment contains a missing version")
    return environment


def run_tests(root=ROOT):
    """Run the full suite once and return its passed-test count."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=root,
        text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    if completed.returncode:
        sys.stderr.write(output)
        raise RuntimeError("pytest failed; report evidence was not regenerated")
    matches = re.findall(r"(\d+) passed", output)
    if not matches:
        raise RuntimeError("could not parse passed-test count from pytest output")
    print(output.strip())
    return int(matches[-1])


def source_revision(root=ROOT):
    override = os.environ.get("MULTIPGS_REPORT_SOURCE_REVISION")
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=root,
        text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def build_evidence(test_count, root=ROOT):
    inputs = report_inputs(root)
    derived = benchmark_evidence(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": package_version(root),
        "source_revision": source_revision(root),
        "input_digest_sha256": report_input_digest(root, inputs),
        "inputs": [path.as_posix() for path in inputs],
        "tests": test_environment(test_count),
        **derived,
    }


_MACROS = {
    "fit_replicates": "ReportFitReplicates",
    "fit_wins": "ReportFitWins",
    "fit_uplift_min": "ReportFitUpliftMin",
    "fit_uplift_max": "ReportFitUpliftMax",
    "fit_gap_closed_min_pct": "ReportFitGapClosedMinPct",
    "fit_gap_closed_max_pct": "ReportFitGapClosedMaxPct",
    "fit_cv_minus_holdout": "ReportFitCvMinusHoldout",
    "null_passes": "ReportNullPasses",
    "null_replicates": "ReportNullReplicates",
    "sumstat_beta_correlation": "ReportSumstatBetaCorrelation",
    "sumstat_r2_min": "ReportSumstatRtwoMin",
    "sumstat_r2_max": "ReportSumstatRtwoMax",
    "pumas_gaussian_error_pct": "ReportPumasGaussianErrorPct",
    "pumas_binary_error_pct": "ReportPumasBinaryErrorPct",
    "ld_true_drop_n50": "ReportLdTrueDropNfifty",
    "ld_report_inflation_n50": "ReportLdInflationNfifty",
    "ld_report_inflation_n50_pct": "ReportLdInflationNfiftyPct",
    "overlap_multi_ratio_010": "ReportOverlapMultiRatioTen",
    "overlap_single_ratio_010": "ReportOverlapSingleRatioTen",
    "overlap_multi_ratio_025": "ReportOverlapMultiRatioTwentyFive",
    "overlap_single_ratio_025": "ReportOverlapSingleRatioTwentyFive",
    "overlap_detected_nonnull": "ReportOverlapDetectedNonnull",
    "overlap_nonnull_replicates": "ReportOverlapNonnullReplicates",
    "overlap_detected_null": "ReportOverlapDetectedNull",
    "overlap_null_replicates": "ReportOverlapNullReplicates",
    "real_meta_expected_r2": "ReportRealMetaExpectedRtwo",
    "real_meta_decorrelated_expected_r2": "ReportRealMetaDecorrelatedRtwo",
}


def render_generated_tex(evidence):
    """Render the deterministic TeX macros consumed by the report template."""
    tests = evidence["tests"]
    digest_chunks = r"\allowbreak{}".join(
        evidence["input_digest_sha256"][start:start + 8]
        for start in range(0, 64, 8))
    lines = [
        "% Generated by report/generate_evidence.py; do not edit.",
        f"\\newcommand{{\\ReportPackageVersion}}{{{evidence['package_version']}}}",
        f"\\newcommand{{\\ReportSourceRevision}}{{{evidence['source_revision']}}}",
        f"\\newcommand{{\\ReportInputDigest}}{{{evidence['input_digest_sha256']}}}",
        f"\\newcommand{{\\ReportInputDigestDisplay}}{{\\texttt{{{digest_chunks}}}}}",
        f"\\newcommand{{\\ReportTestCount}}{{{tests['count']}}}",
        f"\\newcommand{{\\ReportPythonVersion}}{{{tests['python']}}}",
        f"\\newcommand{{\\ReportNumpyVersion}}{{{tests['numpy']}}}",
        f"\\newcommand{{\\ReportScikitLearnVersion}}{{{tests['scikit_learn']}}}",
        f"\\newcommand{{\\ReportPytestVersion}}{{{tests['pytest']}}}",
        f"\\newcommand{{\\ReportLdpredThreeVersion}}{{{tests['ldpred3']}}}",
    ]
    for key, macro in _MACROS.items():
        lines.append(f"\\newcommand{{\\{macro}}}{{{evidence['headlines'][key]}}}")
    return "\n".join(lines) + "\n"


def write_evidence(evidence, root=ROOT):
    root = Path(root)
    (root / "report" / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "report" / "generated_evidence.tex").write_text(
        render_generated_tex(evidence), encoding="utf-8")


def _prior_test_count(root=ROOT):
    """Return a positive prior count for the regeneration bootstrap pass."""
    try:
        evidence = json.loads(
            (Path(root) / "report" / "evidence.json").read_text(
                encoding="utf-8"))
        count = int(evidence["tests"]["count"])
    except (FileNotFoundError, KeyError, TypeError, ValueError,
            json.JSONDecodeError):
        return 1
    return max(count, 1)


def _normalized_tex(text):
    return re.sub(r"\s+", "", text)


def validate_report_source(evidence, root=ROOT):
    """Validate template wiring, generated headlines, and plotted values."""
    text = (Path(root) / "report" / "multipgs_methods.tex").read_text(
        encoding="utf-8")
    if r"\input{generated_evidence.tex}" not in text:
        raise AssertionError("methods report does not input generated evidence")
    required_macros = (
        "ReportPackageVersion", "ReportInputDigest", "ReportSourceRevision",
        "ReportTestCount", *_MACROS.values())
    missing = [macro for macro in required_macros if f"\\{macro}" not in text]
    if missing:
        raise AssertionError(f"methods report omits evidence macros: {missing}")
    normalized = _normalized_tex(text)
    for label, coordinates in evidence["plots"].items():
        expected = _normalized_tex(f"coordinates {{{coordinates}}};")
        if expected not in normalized:
            raise AssertionError(f"methods report plot is stale: {label}")


def validate_committed_evidence(root=ROOT):
    """Read-only freshness check; deliberately does not execute pytest."""
    root = Path(root)
    evidence = json.loads((root / "report" / "evidence.json").read_text(
        encoding="utf-8"))
    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("unsupported report evidence schema")
    expected_inputs = [path.as_posix() for path in report_inputs(root)]
    if evidence.get("inputs") != expected_inputs:
        raise AssertionError("declared report input list is stale")
    digest = report_input_digest(root, map(Path, expected_inputs))
    if evidence.get("input_digest_sha256") != digest:
        raise AssertionError(
            "report inputs changed; run report/generate_evidence.py and rebuild PDF")
    if evidence.get("package_version") != package_version(root):
        raise AssertionError("report package version is stale")
    if not re.fullmatch(r"[0-9a-f]{7,40}", evidence.get("source_revision", "")):
        raise AssertionError("report source_revision is not a Git revision")
    tests = evidence.get("tests", {})
    if tests.get("status") != "passed" or int(tests.get("count", 0)) <= 0:
        raise AssertionError("report test metadata does not record a passing suite")
    for name in ("python", "numpy", "scikit_learn", "pytest", "ldpred3"):
        value = tests.get(name)
        if not isinstance(value, str) or not value.strip():
            raise AssertionError(f"report test metadata omits {name} version")
    expected_benchmarks = benchmark_evidence(root)
    for key in ("headlines", "plots"):
        if evidence.get(key) != expected_benchmarks[key]:
            raise AssertionError(f"report {key} are stale")
    generated = (root / "report" / "generated_evidence.tex").read_text(
        encoding="utf-8")
    if generated != render_generated_tex(evidence):
        raise AssertionError("report/generated_evidence.tex is stale")
    validate_report_source(evidence, root)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-count", type=int,
        help="record an already verified count instead of running pytest")
    parser.add_argument(
        "--check", action="store_true",
        help="check committed evidence without running tests or writing files")
    args = parser.parse_args(argv)
    if args.check:
        validate_committed_evidence(ROOT)
        print("report evidence is current")
        return 0
    if args.test_count is not None and args.test_count <= 0:
        parser.error("--test-count must be positive")
    if args.test_count is not None:
        count = args.test_count
    else:
        # Freshness tests run inside the full suite. Write a current bootstrap
        # manifest first, then replace its prior count with the count just
        # observed. Generated outputs are excluded from the input digest.
        old_evidence = EVIDENCE_PATH.read_bytes() if EVIDENCE_PATH.exists() else None
        old_tex = (GENERATED_TEX_PATH.read_bytes()
                   if GENERATED_TEX_PATH.exists() else None)
        write_evidence(build_evidence(_prior_test_count(ROOT), ROOT), ROOT)
        try:
            count = run_tests(ROOT)
        except Exception:
            if old_evidence is None:
                EVIDENCE_PATH.unlink(missing_ok=True)
            else:
                EVIDENCE_PATH.write_bytes(old_evidence)
            if old_tex is None:
                GENERATED_TEX_PATH.unlink(missing_ok=True)
            else:
                GENERATED_TEX_PATH.write_bytes(old_tex)
            raise
    evidence = build_evidence(count, ROOT)
    write_evidence(evidence, ROOT)
    validate_committed_evidence(ROOT)
    print(f"wrote report evidence for {count} passing tests")
    print(f"input sha256: {evidence['input_digest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

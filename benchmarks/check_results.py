#!/usr/bin/env python3
"""Check committed benchmark summaries, provenance, and copied README values.

The benchmark README is intentionally readable rather than generated.  This
small check keeps that choice honest: summaries are recomputed from raw rows,
and the principal tables are compared with their checked CSV sources.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _rows(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    return float(value)


def _close(actual, expected, label):
    if math.isnan(actual) and math.isnan(expected):
        return
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _check_summary(stem, keys, derived=None):
    raw = _rows(f"{stem}.csv")
    summary = _rows(f"{stem}_summary.csv")
    derived = derived or {}
    for item in summary:
        selected = [row for row in raw
                    if all(row[key] == item[key] for key in keys)]
        if not selected:
            raise AssertionError(f"{stem}: summary group has no raw rows: {item}")
        if "n_seeds" in item and int(item["n_seeds"]) != len(selected):
            raise AssertionError(f"{stem}: n_seeds does not match raw rows")
        for column, value in item.items():
            if not column.endswith("_mean"):
                continue
            field = column[:-5]
            if field in derived:
                values = [derived[field](row) for row in selected]
            elif field in selected[0]:
                values = [_number(row[field]) for row in selected]
            else:
                continue
            values = [x for x in values if math.isfinite(x)]
            expected_mean = statistics.fmean(values)
            _close(_number(value), expected_mean, f"{stem}:{column}")
            sd_column = f"{field}_sd"
            if sd_column in item:
                expected_sd = statistics.stdev(values) if len(values) > 1 else math.nan
                _close(_number(item[sd_column]), expected_sd,
                       f"{stem}:{sd_column}")


def _check_provenance():
    required = {"generated_utc", "elapsed_seconds", "command", "platform",
                "python", "multipgs", "numpy"}
    for path in sorted(RESULTS.glob("*_provenance.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - data.keys()
        if missing:
            raise AssertionError(f"{path.name}: missing {sorted(missing)}")
        stem = path.name.removesuffix("_provenance.json")
        if not (HERE / f"{stem}.py").exists():
            raise AssertionError(f"{path.name}: no producer benchmarks/{stem}.py")
        source = data.get("source")
        if source is not None:
            source_required = {"repository_commit", "source_dirty",
                               "dirty_scope", "script", "script_sha256"}
            missing = source_required - source.keys()
            if missing:
                raise AssertionError(
                    f"{path.name}: incomplete source identity {sorted(missing)}")


def _check_readme():
    text = (HERE / "README.md").read_text(encoding="utf-8")
    normalized = text.replace("−", "-")

    calibration, = _rows("sumstat_calibration_summary.csv")
    calibration_rows = (
        f"| Gram identity, maximum absolute error | "
        f"{_number(calibration['identity_gram_max_abs_mean']):.2e} |",
        f"| Cross-moment identity, maximum absolute error | "
        f"{_number(calibration['identity_c_max_abs_mean']):.2e} |",
        f"| Null tuning MSE | "
        f"{_number(calibration['null_tuning_mse_mean']):.6f} |",
        f"| Null untouched-assessment MSE | "
        f"{_number(calibration['null_assessment_mse_mean']):.6f} |",
        f"| Gaussian plug-in covariance relative error | "
        f"{_number(calibration['gaussian_plugin_cov_rel_error_mean']):.4f} |",
        f"| Binary plug-in covariance relative error | "
        f"{_number(calibration['binary_plugin_cov_rel_error_mean']):.4f} |",
    )
    for expected in calibration_rows:
        if expected not in normalized:
            raise AssertionError(f"sumstat_calibration README row is stale: {expected}")

    for row in _rows("fit_accuracy_summary.csv"):
        expected = (f"| {int(row['n']):,} | {row['n_scores']} | {row['h2']} | "
                    f"{_number(row['best_single_r2_mean']):.3f} | "
                    f"{_number(row['multi_r2_mean']):.3f} | "
                    f"{_number(row['oracle_r2_mean']):.3f} | "
                    f"{_number(row['uplift_mean']):+.3f} |")
        if expected not in text:
            raise AssertionError(f"fit_accuracy README row is stale: {expected}")

    for row in _rows("null_gate_summary.csv"):
        expected = (
            f"| {int(row['n']):,} | "
            f"{_number(row['null_model_returned_mean']):.3f} | "
            f"{_number(row['null_cv_r2_mean']):.3f} | "
            f"{_number(row['signal_cv_minus_heldout_mean']):+.3f} ± "
            f"{_number(row['signal_cv_minus_heldout_sd']):.3f} |")
        if expected not in normalized:
            raise AssertionError(f"null_gate README row is stale: {expected}")

    agreement, = _rows("sumstat_vs_individual_summary.csv")
    agreement_rows = {
        "`beta_std` correlation, individual vs summary fit":
            agreement["agreement_beta_std_corr_mean"],
        "Held-out R², individual-level CMSA":
            agreement["individual_holdout_r2_mean"],
        "Held-out R², summary fit tuned independently":
            agreement["sumstat_holdout_r2_independent_mean"],
        "Held-out R², summary fit with PUMAS pseudotuning":
            agreement["sumstat_holdout_r2_pumas_mean"],
        "Held-out R², summary fit tuned in-sample":
            agreement["sumstat_holdout_r2_none_mean"],
    }
    for label, value in agreement_rows.items():
        expected = f"| {label} | {_number(value):.4f} |"
        if expected not in text:
            raise AssertionError(
                f"sumstat_vs_individual README row is stale: {expected}")

    labels = {
        "best_single_max_n_eff": "best single by maximum `n_eff`",
        "equal_weight": "equal weight",
        "sqrt_n_eff": "`sqrt_n_eff`",
        "expected_r2": "`expected_r2`",
        "decorrelated_n_eff": "decorrelated `n_eff`",
        "decorrelated_expected_r2": "decorrelated `expected_r2`",
        "best_single_oracle": "best single selected on evaluation moments",
    }
    for row in _rows("real_meta_rules.csv"):
        expected = f"| {labels[row['rule']]} | {_number(row['r2']):.5f} |"
        if expected not in text:
            raise AssertionError(f"real_meta_rules README row is stale: {expected}")
    real_meta = {row["rule"]: _number(row["r2"])
                 for row in _rows("real_meta_rules.csv")}
    degradation = (real_meta["expected_r2"]
                   / real_meta["decorrelated_expected_r2"])
    if f"{degradation:.1f} times worse" not in text:
        raise AssertionError("real_meta_rules README degradation is stale")

    real_ld = {row["reference_n"]: row
               for row in _rows("real_ld_simulation_summary.csv")}
    full, small = real_ld["inf"], real_ld["50"]
    fragments = (
        f"{_number(full['true_r2_multi_mean']):.5f} to "
        f"{_number(small['true_r2_multi_mean']):.5f}",
        f"reports {_number(small['regime_a_r2_own_reference_mean']):.5f}",
        f"inflation of {_number(small['own_reference_minus_true_mean']):.5f} R²",
        f"{100 * _number(small['own_reference_minus_true_mean']) / _number(small['true_r2_multi_mean']):.1f}% relative",
    )
    for fragment in fragments:
        if fragment not in text:
            raise AssertionError(f"real-LD README value is stale: {fragment}")

    for stem in ("overlap_inflation", "real_ld_simulation", "real_meta_rules"):
        version = json.loads(
            (RESULTS / f"{stem}_provenance.json").read_text(encoding="utf-8")
        )["ldpred3"]
        if stem not in text or f"ldpred3={version}" not in text:
            raise AssertionError(f"README omits historical provenance caveat: {stem}")

    pyproject = (HERE.parent / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"(ldpred3[^\"]+)"', pyproject)
    if match is None or match.group(1) not in text:
        raise AssertionError("README supported ldpred3 range is stale")


def _check_report_evidence():
    path = HERE.parent / "report" / "generate_evidence.py"
    spec = importlib.util.spec_from_file_location("report_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_committed_evidence(HERE.parent)


def check_results():
    """Raise ``AssertionError`` if a committed benchmark artifact has drifted."""
    _check_summary("fit_accuracy", ("n", "n_scores", "h2"), {
        "uplift": lambda row: (_number(row["multi_r2"])
                               - _number(row["best_single_r2"])),
    })
    _check_summary("meta_rules", ("shared_error_correlation",))
    _check_summary("null_gate", ("n",))
    _check_summary("overlap_inflation", ("overlap", "n_variants"))
    _check_summary("real_ld_simulation", ("reference_n",))
    _check_summary("sumstat_calibration", ())
    _check_summary("sumstat_vs_individual", ())
    _check_provenance()
    _check_readme()
    _check_report_evidence()


if __name__ == "__main__":
    check_results()
    print("benchmark artifacts are internally consistent")

#!/usr/bin/env python3
"""Calibrate the summary-statistic contracts against simulated individuals.

The benchmark records three quantities that unit tests alone cannot establish:
the sufficient-statistic identity on discrete dosages, selection optimism under
the null, and the error of the joint-Gaussian covariance used for PUMAS-style
pseudotuning. Raw per-seed rows, a summary, and runtime provenance are written
together.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import multipgs
from multipgs import multi_pgs_sumstats, score_moments


FIELDS = (
    "identity_gram_max_abs",
    "identity_c_max_abs",
    "null_tuning_mse",
    "null_assessment_mse",
    "null_assessment_minus_tuning_mse",
    "null_n_selected",
    "gaussian_plugin_cov_rel_error",
    "binary_plugin_cov_rel_error",
)


def _version(name):
    try:
        value = getattr(importlib.import_module(name), "__version__", None)
        if value is not None:
            return str(value)
    except ImportError:
        pass
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _standardize(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _identity(seed, n, m, k):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.05, 0.5, size=m)
    dosage = rng.binomial(2, af, size=(n, m)).astype(float)
    dosage -= dosage.mean(axis=0)
    dosage_sd = dosage.std(axis=0)
    x = dosage / dosage_sd
    raw_w = np.zeros((m, k))
    for j in range(k):
        idx = rng.choice(m, size=min(12, m), replace=False)
        raw_w[idx, j] = rng.normal(size=idx.size)
    # The Catalog-style weights multiply allele counts. The corresponding
    # standardized-genotype weights are dataset-specific: diag(sd) @ raw_w.
    w = dosage_sd[:, None] * raw_w
    scores = dosage @ raw_w
    y = scores[:, 0] / scores[:, 0].std() + rng.normal(size=n)
    y = (y - y.mean()) / y.std()
    ld = x.T @ x / n
    z = x.T @ y / n
    c, gram, _ = score_moments(w, z, ld, weights_gwas=w)
    return (float(np.max(np.abs(gram - scores.T @ scores / n))),
            float(np.max(np.abs(c - scores.T @ y / n))))


def _null_selection(seed, n, k, n_lambda):
    rng = np.random.default_rng(seed + 100_000)
    sd = 1.0 / np.sqrt(n)
    z_train = rng.normal(scale=sd, size=k)
    z_tune = rng.normal(scale=sd, size=k)
    z_test = rng.normal(scale=sd, size=k)
    identity = np.eye(k)
    fit = multi_pgs_sumstats(
        identity, z_train, identity, weights_gwas=identity,
        z_valid=z_tune, ld_valid=identity,
        weights_gwas_valid=identity, weights_ld_valid=identity,
        tune="independent", n_lambda=n_lambda, var_y=1.0)
    assessment = fit.evaluate(z_test, identity, var_y=1.0, regime="A")
    tuning = float(fit.selection_mse)
    test = float(assessment.mse)
    return tuning, test, test - tuning, fit.n_selected


def _plugin_error(scores, y):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    scores = scores - scores.mean(axis=0)
    y = y - y.mean()
    n = y.size
    gram = scores.T @ scores / n
    c = scores.T @ y / n
    products = scores * y[:, None]
    products -= products.mean(axis=0)
    observed = products.T @ products / n
    plugin = y.var() * gram + np.outer(c, c)
    return float(np.linalg.norm(plugin - observed) /
                 max(np.linalg.norm(observed), 1e-300))


def _plugin_models(seed, n, k):
    rng = np.random.default_rng(seed + 200_000)
    a = rng.normal(size=(k, k))
    scores = rng.normal(size=(n, k)) @ a
    beta = rng.normal(size=k)
    signal = scores @ beta
    y_gaussian = signal / signal.std() + rng.normal(size=n)
    gaussian = _plugin_error(scores, y_gaussian)

    m = max(4 * k, 40)
    af = rng.uniform(0.05, 0.5, size=m)
    dosage = _standardize(rng.binomial(2, af, size=(n, m)))
    w = rng.normal(size=(m, k)) / np.sqrt(m)
    score_binary = dosage @ w
    linear = score_binary[:, :min(3, k)].sum(axis=1)
    probability = 1.0 / (1.0 + np.exp(-linear / linear.std()))
    y_binary = rng.binomial(1, probability, size=n)
    binary = _plugin_error(score_binary, y_binary)
    return gaussian, binary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--variants", type=int, default=80)
    parser.add_argument("--scores", type=int, default=20)
    parser.add_argument("--null-scores", type=int, default=80)
    parser.add_argument("--n-lambda", type=int, default=60)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)
    if min(args.seeds, args.n, args.variants, args.scores,
           args.null_scores, args.n_lambda) < 1:
        parser.error("all numeric arguments must be positive")

    started = time.perf_counter()
    rows = []
    for seed in range(args.seeds):
        gram_error, c_error = _identity(
            seed, args.n, args.variants, args.scores)
        tuning, assessment, optimism, n_selected = _null_selection(
            seed, args.n, args.null_scores, args.n_lambda)
        gaussian, binary = _plugin_models(seed, args.n, min(args.scores, 12))
        rows.append({
            "seed": seed,
            "identity_gram_max_abs": gram_error,
            "identity_c_max_abs": c_error,
            "null_tuning_mse": tuning,
            "null_assessment_mse": assessment,
            "null_assessment_minus_tuning_mse": optimism,
            "null_n_selected": n_selected,
            "gaussian_plugin_cov_rel_error": gaussian,
            "binary_plugin_cov_rel_error": binary,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "sumstat_calibration.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", *FIELDS),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {"n_seeds": len(rows)}
    for field in FIELDS:
        values = np.asarray([row[field] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary[f"{field}_mean"] = (float(finite.mean()) if finite.size
                                             else float("nan"))
        summary[f"{field}_sd"] = (float(finite.std(ddof=1))
                                   if finite.size > 1 else float("nan"))
    summary_path = args.output_dir / "sumstat_calibration_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "command": ([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
                    if argv is None else
                    [sys.executable, str(Path(__file__).resolve()), *argv]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "multipgs": multipgs.__version__,
        "numpy": np.__version__,
        "ldpred3": _version("ldpred3"),
        "numba": _version("numba"),
        "parameters": vars(args) | {"output_dir": str(args.output_dir)},
        "note": ("The PUMAS covariance rows measure the Gaussian plug-in "
                 "against empirical score-by-phenotype fourth moments; they "
                 "are calibration diagnostics, not accuracy estimates."),
    }
    with (args.output_dir / "sumstat_calibration_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

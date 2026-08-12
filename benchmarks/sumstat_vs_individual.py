#!/usr/bin/env python3
"""Compare the summary-statistic route against individuals, and tuning rules.

Unit tests check that :func:`multipgs.multi_pgs_sumstats` runs and honours its
input contracts; they cannot establish two practical facts: whether a
combination learned from moments alone agrees with the individual-level CMSA
fit computed on the same cohort, and what pseudotuning (PUMAS) or in-sample
tuning costs relative to an independent tuning GWAS. On discrete dosages
shared across three independent cohorts — one for training moments, one for
tuning, one for held-out assessment — this benchmark records the coefficient
correlation and held-out R² of both routes, and held-out R² under all three
tuning regimes. Raw per-seed rows, a summary, and runtime provenance are
written together.
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

# Measure the checkout beside this artifact even when no editable install is
# active. This also keeps version provenance tied to the code being exercised.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import multipgs
from benchmarks._provenance import benchmark_identity
from multipgs import multi_pgs_fit, multi_pgs_sumstats, r2


FIELDS = (
    "agreement_beta_std_corr",
    "individual_holdout_r2",
    "sumstat_holdout_r2_independent",
    "sumstat_holdout_r2_pumas",
    "sumstat_holdout_r2_none",
    "selection_mse_independent",
    "selection_mse_pumas",
    "selection_mse_none",
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


def _cohort(rng, n, af, raw_w, signal):
    """One cohort's moments and raw scores, on shared variants and weights."""
    m = af.size
    dosage = rng.binomial(2, af, size=(n, m)).astype(float)
    dosage -= dosage.mean(axis=0)
    dosage_sd = dosage.std(axis=0)
    x = dosage / dosage_sd
    # The Catalog-style weights multiply allele counts. The corresponding
    # standardized-genotype weights are dataset-specific: diag(sd) @ raw_w.
    w = dosage_sd[:, None] * raw_w
    scores = dosage @ raw_w
    g = scores[:, :signal.size] @ signal
    y = g / g.std() + rng.normal(size=n)
    y = (y - y.mean()) / y.std()
    ld = x.T @ x / n
    z = x.T @ y / n
    return w, z, ld, scores, y


def _corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() <= 0.0 or b.std() <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _replicate(seed, n, m, k, n_signal, fit_kwargs, n_lambda):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.05, 0.5, size=m)
    raw_w = np.zeros((m, k))
    for j in range(k):
        idx = rng.choice(m, size=min(12, m), replace=False)
        raw_w[idx, j] = rng.normal(size=idx.size)
    # Three independent cohorts on the same variants and component weights:
    # A fits, B tunes, C assesses.
    signal = np.linspace(1.0, 0.6, n_signal)
    w_a, z_a, ld_a, scores_a, y_a = _cohort(rng, n, af, raw_w, signal)
    w_b, z_b, ld_b, _, _ = _cohort(rng, n, af, raw_w, signal)
    _, _, _, scores_c, y_c = _cohort(rng, n, af, raw_w, signal)

    # Agreement arm: individual-level CMSA vs moments-only combination, both
    # learned from cohort A and selected with cohort B as tuning data.
    ind_fit = multi_pgs_fit(scores_a, y_a, seed=seed, **fit_kwargs)
    sum_fit = multi_pgs_sumstats(
        w_a, z_a, ld_a, weights_gwas=w_a, alpha=1.0, n_lambda=n_lambda,
        tune="independent", z_valid=z_b, ld_valid=ld_b,
        weights_gwas_valid=w_b, weights_ld_valid=w_b)

    # Tuning arm: identical training moments, three selection regimes.
    pumas_fit = multi_pgs_sumstats(
        w_a, z_a, ld_a, weights_gwas=w_a, alpha=1.0, n_lambda=n_lambda,
        tune="pumas", n_eff=n, weights_independent_of_z=True)
    none_fit = multi_pgs_sumstats(
        w_a, z_a, ld_a, weights_gwas=w_a, alpha=1.0, n_lambda=n_lambda,
        tune="none")

    return {"seed": seed,
            "agreement_beta_std_corr": _corr(ind_fit.beta_std,
                                             sum_fit.beta_std),
            "individual_holdout_r2": r2(y_c, ind_fit.multi_pgs(scores_c)),
            "sumstat_holdout_r2_independent": r2(y_c,
                                                 sum_fit.multi_pgs(scores_c)),
            "sumstat_holdout_r2_pumas": r2(y_c,
                                           pumas_fit.multi_pgs(scores_c)),
            "sumstat_holdout_r2_none": r2(y_c, none_fit.multi_pgs(scores_c)),
            "selection_mse_independent": float(sum_fit.selection_mse),
            "selection_mse_pumas": float(pumas_fit.selection_mse),
            "selection_mse_none": float(none_fit.selection_mse)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n", type=int, default=8_000)
    parser.add_argument("--variants", type=int, default=80)
    parser.add_argument("--scores", type=int, default=20)
    parser.add_argument("--signal-scores", type=int, default=3)
    parser.add_argument("--fit-folds", type=int, default=5)
    parser.add_argument("--fit-assessment-folds", type=int, default=3)
    parser.add_argument("--fit-n-lambda", type=int, default=40)
    parser.add_argument("--n-lambda", type=int, default=60)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)
    if min(args.seeds, args.n, args.variants, args.scores, args.signal_scores,
           args.fit_folds, args.fit_assessment_folds, args.fit_n_lambda,
           args.n_lambda) < 1 or args.signal_scores > args.scores:
        parser.error("numeric arguments are too small for this design")

    fit_kwargs = {"n_folds": args.fit_folds,
                  "assessment_folds": args.fit_assessment_folds,
                  "n_lambda": args.fit_n_lambda}
    started = time.perf_counter()
    rows = [_replicate(seed, args.n, args.variants, args.scores,
                       args.signal_scores, fit_kwargs, args.n_lambda)
            for seed in range(args.seeds)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "sumstat_vs_individual.csv"
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
    summary_path = args.output_dir / "sumstat_vs_individual_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)

    provenance = {
        "source": benchmark_identity(__file__),
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
        "note": ("The moments are computed from the same individuals the "
                 "individual-level fit trains on, so agreement measures "
                 "selection-routing differences (nested CMSA vs a tuning "
                 "GWAS), not approximation error of the sufficient "
                 "statistics. The tuning arm quantifies the cost of "
                 "pseudotuning and in-sample tuning relative to an "
                 "independent tuning GWAS; cohort C stays untouched by "
                 "fitting, tuning and selection."),
    }
    with (args.output_dir / "sumstat_vs_individual_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

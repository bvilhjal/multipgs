#!/usr/bin/env python3
"""Calibrate the nested signal gate of :func:`multipgs.multi_pgs_fit`.

Unit tests exercise the gate on a single example; they cannot establish its
two operating characteristics: how often the penalized model passes when no
score carries signal at all, and how far the internal nested-CV estimate sits
from held-out accuracy when signal is present. Over repeated simulated
panels, a null arm replaces the phenotype with fresh noise and records the
null-model return rate, while a signal arm compares ``cv_r2`` on the
training split with true held-out R² of the returned fit. Raw per-seed rows,
a per-regime summary, and runtime provenance are written together.
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
from multipgs import multi_pgs_fit, r2, simulate_panel


FIELDS = (
    "null_model_returned",
    "null_cv_r2",
    "null_n_selected",
    "signal_null_model",
    "signal_cv_r2",
    "signal_heldout_r2",
    "signal_cv_minus_heldout",
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


def _replicate(n, k, h2, n_causal, seed, fit_kwargs):
    sim = simulate_panel(n=n, n_scores=k, n_causal=n_causal, h2=h2, seed=seed)

    # Null arm: same scores and covariates, but the phenotype is fresh noise,
    # so no score can carry signal. Fitted on all rows; there is nothing
    # worth a held-out split.
    y_null = np.random.default_rng(seed + 100_000).normal(size=n)
    null_fit = multi_pgs_fit(sim.scores, y_null, covar=sim.covar,
                             score_ids=sim.score_ids, seed=seed, **fit_kwargs)
    null_cv_r2 = null_fit.cv_r2

    # Signal arm: the simulator's real phenotype, fitted on a 75% split so
    # the returned model has a genuinely untouched assessment.
    rng = np.random.default_rng(seed + 200_000)
    order = rng.permutation(n)
    n_train = (3 * n) // 4
    tr, te = order[:n_train], order[n_train:]
    fit = multi_pgs_fit(sim.scores[tr], sim.y[tr], covar=sim.covar[tr],
                        score_ids=sim.score_ids, seed=seed, **fit_kwargs)
    heldout = r2(sim.y[te], fit.multi_pgs(sim.scores[te]))
    signal_cv_r2 = fit.cv_r2
    signal_cv_r2 = (float(signal_cv_r2) if signal_cv_r2 is not None
                    else float("nan"))

    return {"n": n, "seed": seed,
            "null_model_returned": int("null_model" in null_fit.log),
            "null_cv_r2": (float(null_cv_r2) if null_cv_r2 is not None
                           else float("nan")),
            "null_n_selected": null_fit.n_selected,
            "signal_null_model": int("null_model" in fit.log),
            "signal_cv_r2": signal_cv_r2,
            "signal_heldout_r2": heldout,
            "signal_cv_minus_heldout": signal_cv_r2 - heldout}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n", type=int, nargs="+", default=(1_000, 5_000))
    parser.add_argument("--scores", type=int, default=50)
    parser.add_argument("--n-causal", type=int, default=8)
    parser.add_argument("--h2", type=float, default=0.4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--assessment-folds", type=int, default=3)
    parser.add_argument("--n-lambda", type=int, default=40)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)
    if min(args.seeds, args.scores, args.n_causal, args.folds,
           args.assessment_folds, args.n_lambda) < 1 or min(args.n) < 4:
        parser.error("numeric arguments are too small for this design")

    fit_kwargs = {"n_folds": args.folds,
                  "assessment_folds": args.assessment_folds,
                  "n_lambda": args.n_lambda}
    started = time.perf_counter()
    rows = []
    for n in args.n:
        for seed in range(args.seeds):
            rows.append(_replicate(n, args.scores, args.h2, args.n_causal,
                                   seed, fit_kwargs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "null_gate.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n",
                                fieldnames=("n", "seed", *FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for n in args.n:
        selected = [row for row in rows if row["n"] == n]
        item = {"n": n, "n_seeds": len(selected)}
        for field in FIELDS:
            values = np.asarray([row[field] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            item[f"{field}_mean"] = (float(finite.mean()) if finite.size
                                     else float("nan"))
            item[f"{field}_sd"] = (float(finite.std(ddof=1))
                                   if finite.size > 1 else float("nan"))
        summary.append(item)
    summary_path = args.output_dir / "null_gate_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

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
        "note": ("In the null arm no score carries signal, so the "
                 "null_model return rate is the gate's false-pass control "
                 "(one minus that rate is the false-pass rate). The "
                 "signal-arm cv_r2 minus held-out R² is the bias of the "
                 "internal nested-CV estimate — documented as possibly "
                 "conservative — not a generalization guarantee."),
    }
    with (args.output_dir / "null_gate_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

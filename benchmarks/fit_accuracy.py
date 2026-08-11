#!/usr/bin/env python3
"""Measure end-to-end accuracy of the learned multi-PGS combination.

Unit tests establish that the fit runs and recovers support on one small
panel; they cannot establish the package's headline claim — that the learned
combination beats the best single input score on untouched individuals. This
benchmark fits :func:`multipgs.multi_pgs_fit` on a 75% training split over a
grid of cohort sizes, panel sizes and heritabilities, and records held-out
R² of the multi-PGS, of the best single score (chosen on the training
split), and of the genetic-value oracle, alongside the internal nested-CV
estimate and support recovery. Raw per-seed rows, a per-regime summary, and
runtime provenance are written together.

Replicates are independent, so they run in worker processes to keep the
full grid within a couple of minutes; results are written in grid order
regardless of completion order.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Worker processes must not multiply BLAS threads: one process per core,
# each spawning its own OpenBLAS thread pool, oversubscribes the machine and
# slows every fit by an order of magnitude. Set before numpy is imported, in
# the parent and in every spawned worker that re-imports this module; an
# explicit user setting wins.
for _threads in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import numpy as np

# Measure the checkout beside this artifact even when no editable install is
# active. This also keeps version provenance tied to the code being exercised.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import multipgs
from multipgs import multi_pgs_fit, r2, simulate_panel


FIELDS = (
    "multi_r2",
    "best_single_r2",
    "oracle_r2",
    "cv_r2",
    "null_model",
    "n_selected",
    "support_precision",
    "support_recall",
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


def _replicate(task):
    """One train/test replicate; a top-level function so workers can run it."""
    n, k, h2, seed, n_causal, fit_kwargs = task
    sim = simulate_panel(n=n, n_scores=k, n_causal=n_causal, h2=h2, seed=seed)
    rng = np.random.default_rng(seed + 300_000)
    order = rng.permutation(n)
    n_train = (3 * n) // 4
    tr, te = order[:n_train], order[n_train:]

    fit = multi_pgs_fit(sim.scores[tr], sim.y[tr], covar=sim.covar[tr],
                        score_ids=sim.score_ids, seed=seed, **fit_kwargs)
    multi_r2 = r2(sim.y[te], fit.multi_pgs(sim.scores[te]))

    train_r2 = [r2(sim.y[tr], sim.scores[tr, j]) for j in range(k)]
    best_single_r2 = r2(sim.y[te], sim.scores[te, int(np.argmax(train_r2))])
    oracle_r2 = r2(sim.y[te], sim.genetic_value[te])

    true_support = sim.beta_true != 0.0
    selected = fit.beta != 0.0
    hits = int(np.count_nonzero(selected & true_support))
    n_selected = int(np.count_nonzero(selected))
    precision = hits / n_selected if n_selected else float("nan")
    recall = hits / int(np.count_nonzero(true_support))

    cv_r2 = fit.cv_r2
    return {"n": n, "n_scores": k, "h2": h2, "seed": seed,
            "multi_r2": multi_r2,
            "best_single_r2": best_single_r2,
            "oracle_r2": oracle_r2,
            "cv_r2": float(cv_r2) if cv_r2 is not None else float("nan"),
            "null_model": int("null_model" in fit.log),
            "n_selected": n_selected,
            "support_precision": precision,
            "support_recall": recall}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n", type=int, nargs="+", default=(2_000, 10_000))
    parser.add_argument("--scores", type=int, nargs="+", default=(50, 200))
    parser.add_argument("--h2", type=float, nargs="+", default=(0.2, 0.5))
    parser.add_argument("--n-causal", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--assessment-folds", type=int, default=3)
    parser.add_argument("--n-lambda", type=int, default=40)
    parser.add_argument("--jobs", type=int,
                        default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)
    if min(args.seeds, args.n_causal, args.folds, args.assessment_folds,
           args.n_lambda, args.jobs) < 1 or min(args.n) < 4 \
            or min(args.scores) < args.n_causal:
        parser.error("numeric arguments are too small for this design")

    fit_kwargs = {"n_folds": args.folds,
                  "assessment_folds": args.assessment_folds,
                  "n_lambda": args.n_lambda}
    tasks = [(n, k, h2, seed, args.n_causal, fit_kwargs)
             for n in args.n for k in args.scores for h2 in args.h2
             for seed in range(args.seeds)]

    started = time.perf_counter()
    if args.jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
                max_workers=args.jobs,
                mp_context=mp.get_context("spawn")) as pool:
            rows = list(pool.map(_replicate, tasks, chunksize=1))
    else:
        rows = [_replicate(task) for task in tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "fit_accuracy.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n",
                                fieldnames=("n", "n_scores", "h2", "seed",
                                            *FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for n in args.n:
        for k in args.scores:
            for h2 in args.h2:
                selected = [row for row in rows
                            if row["n"] == n and row["n_scores"] == k
                            and row["h2"] == h2]
                item = {"n": n, "n_scores": k, "h2": h2,
                        "n_seeds": len(selected)}
                metrics = {field: np.asarray([row[field] for row in selected],
                                             dtype=float)
                           for field in FIELDS}
                metrics["uplift"] = (metrics["multi_r2"]
                                     - metrics["best_single_r2"])
                for field, values in metrics.items():
                    finite = values[np.isfinite(values)]
                    item[f"{field}_mean"] = (float(finite.mean())
                                             if finite.size else float("nan"))
                    item[f"{field}_sd"] = (float(finite.std(ddof=1))
                                           if finite.size > 1
                                           else float("nan"))
                summary.append(item)
    summary_path = args.output_dir / "fit_accuracy_summary.csv"
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
        "note": ("All R² values are for the score alone — covariates are "
                 "fitted but excluded from the evaluated prediction, per the "
                 "package's accuracy definition — so the genetic-value oracle "
                 "sits below h2 whenever covariates carry phenotypic "
                 "variance. cv_r2 is the nested-CV estimate on the training "
                 "split only; uplift is multi_r2 minus best_single_r2 per "
                 "replicate. null_model_mean is the rate at which the nested "
                 "gate withheld the penalized model."),
    }
    with (args.output_dir / "fit_accuracy_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

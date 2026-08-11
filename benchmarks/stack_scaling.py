#!/usr/bin/env python3
"""Time representative Gaussian fits in isolated worker processes.

Each case warms the numerical kernels before timing. Peak RSS is the worker's
absolute process peak and therefore includes Python, NumPy, and Numba runtime
state as well as the fit; it is still comparable across rows from one run.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import inspect
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Running this file directly makes ``benchmarks/`` the import root.  Put the
# checkout itself first so both the parent and its worker subprocesses measure
# the code beside this artifact, not an unrelated installed release.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported CI platforms report KiB.
    return value / (1024.0 ** 2 if platform.system() == "Darwin" else 1024.0)


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


def _fit_kwargs(n_folds, assessment_folds, n_lambda):
    from multipgs import multi_pgs_fit

    kwargs = {"n_folds": n_folds, "n_lambda": n_lambda, "seed": 1}
    if "assessment_folds" in inspect.signature(multi_pgs_fit).parameters:
        kwargs["assessment_folds"] = assessment_folds
    return kwargs


def _worker(case, n_folds, assessment_folds, n_lambda):
    from multipgs import multi_pgs_fit, simulate_panel

    n, k = map(int, case.lower().split("x", 1))
    kwargs = _fit_kwargs(n_folds, assessment_folds, n_lambda)
    warm = simulate_panel(n=250, n_scores=12, n_causal=3, seed=0)
    warm_kwargs = dict(kwargs, n_folds=min(n_folds, 3))
    if "assessment_folds" in warm_kwargs:
        warm_kwargs["assessment_folds"] = min(assessment_folds, 3)
    multi_pgs_fit(warm.scores, warm.y, covar=warm.covar, **warm_kwargs)

    sim = simulate_panel(n=n, n_scores=k, n_causal=min(8, k), seed=1)
    started = time.perf_counter()
    fit = multi_pgs_fit(sim.scores, sim.y, covar=sim.covar, **kwargs)
    return {
        "n": n,
        "n_scores": k,
        "n_folds": n_folds,
        "assessment_folds": (assessment_folds
                             if "assessment_folds" in kwargs else 0),
        "n_lambda": n_lambda,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mb": _rss_mb(),
        "cv_r2": fit.cv_r2,
        "n_selected": fit.n_selected,
    }


def _run_case(script, case, args):
    command = [sys.executable, str(script), "--worker", case,
               "--folds", str(args.folds),
               "--assessment-folds", str(args.assessment_folds),
               "--n-lambda", str(args.n_lambda)]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=("2000x50", "5000x100"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--assessment-folds", type=int, default=3)
    parser.add_argument("--n-lambda", type=int, default=40)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results"))
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        print(json.dumps(_worker(args.worker, args.folds,
                                 args.assessment_folds, args.n_lambda)))
        return 0

    script = Path(__file__).resolve()
    started = time.perf_counter()
    rows = [_run_case(script, case, args) for case in args.cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "stack_scaling.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import multipgs
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "command": ([sys.executable, str(script), *sys.argv[1:]]
                    if argv is None else [sys.executable, str(script), *argv]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "multipgs": multipgs.__version__,
        "numpy": np.__version__,
        "ldpred3": _version("ldpred3"),
        "numba": _version("numba"),
        "note": "Peak RSS is absolute worker-process RSS after a warm-up fit.",
    }
    with (args.output_dir / "stack_scaling_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

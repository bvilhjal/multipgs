#!/usr/bin/env python3
"""Reproduce the same-trait meta-PGS comparison in the documentation.

The raw per-seed rows, their mean/standard deviation, and the runtime
environment are written together.  The ``shared`` parameter is an error-term
correlation; it is not a literal fraction of overlapping samples.
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
from multipgs import daetwyler_r2, meta_pgs, r2, simulate_same_trait_panel


METHODS = (
    "best_single",
    "sqrt_n_eff",
    "expected_r2",
    "decorrelated_n_eff",
    "decorrelated_expected_r2",
)


def _score(sim, expected_r2):
    scores = sim.scores
    return {
        "best_single": max(r2(sim.y, scores[:, k])
                           for k in range(scores.shape[1])),
        "sqrt_n_eff": r2(
            sim.y, meta_pgs(scores, n_eff=sim.n_eff).multi_pgs(scores)),
        "expected_r2": r2(
            sim.y, meta_pgs(scores, expected_r2=expected_r2,
                            method="expected_r2").multi_pgs(scores)),
        "decorrelated_n_eff": r2(
            sim.y, meta_pgs(scores, n_eff=sim.n_eff,
                            method="decorrelated").multi_pgs(scores)),
        "decorrelated_expected_r2": r2(
            sim.y, meta_pgs(scores, expected_r2=expected_r2,
                            method="decorrelated").multi_pgs(scores)),
    }


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40_000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--shared", type=float, nargs="+",
                        default=(0.0, 0.3, 0.6, 0.8))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)

    n_eff = np.array([150_000.0, 60_000.0, 20_000.0])
    h2, m_causal, n_variants = 0.4, 5_000, 1_000_000
    expected = daetwyler_r2(h2, m_causal / n_variants, n_eff, n_variants)
    started = time.perf_counter()
    rows = []
    for shared in args.shared:
        for seed in range(args.seeds):
            sim = simulate_same_trait_panel(
                n=args.n, n_eff=n_eff, h2=h2, m_causal=m_causal,
                n_variants=n_variants, shared=shared, seed=seed)
            values = _score(sim, expected)
            rows.append({"shared_error_correlation": shared, "seed": seed,
                         **values})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "meta_rules.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=(
            "shared_error_correlation", "seed", *METHODS))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for shared in args.shared:
        selected = [row for row in rows
                    if row["shared_error_correlation"] == shared]
        item = {"shared_error_correlation": shared, "n_seeds": len(selected)}
        for method in METHODS:
            values = np.array([row[method] for row in selected])
            item[f"{method}_mean"] = float(values.mean())
            item[f"{method}_sd"] = (float(values.std(ddof=1))
                                    if values.size > 1 else float("nan"))
        summary.append(item)
    summary_path = args.output_dir / "meta_rules_summary.csv"
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
        "parameters": {
            "n": args.n, "seeds": args.seeds, "shared": args.shared,
            "n_eff": n_eff.tolist(), "h2": h2,
            "m_causal": m_causal, "n_variants": n_variants,
        },
    }
    with (args.output_dir / "meta_rules_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

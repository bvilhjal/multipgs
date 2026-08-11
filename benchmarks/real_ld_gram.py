#!/usr/bin/env python3
"""Score-space moments on a real genome-wide LD reference.

Every other benchmark here builds its LD from a simulator. That leaves the two
properties which decide whether the summary-statistic route works in practice
unmeasured, because a simulator does not produce them: a real reference is
**block-heterogeneous** — ldpred3 stores small blocks densely and large ones as
low-rank factors — and a real score panel is **rank-deficient and
near-collinear**, because catalog scores for related traits are largely the
same variants reweighted.

This benchmark takes a real reference and reports what
:func:`multipgs.score_gram` and :func:`multipgs.sumstat._validate_moments`
actually see on it:

* the Gram's spectrum on the scale-invariant correlation coordinates, its
  numerical rank, and how many scores come out with no variance at all;
* the cost and peak memory of forming it, split by block representation, since
  the low-rank blocks hold most of the variants and therefore most of the time;
* whether the low-rank fast path in :func:`multipgs.sumstat._block_quadform`
  agrees with routing every block through :func:`ldpred3.ld_matmul`.

The reference is supplied, not shipped: it is a gigabyte-scale artifact with
its own provenance. The one this was developed against is Privé's bigsnpr
HapMap3+ European (UK Biobank) LD, converted to ldpred3's block format by
``ldpred3/benchmarks/convert_bigsnpr_ldref.py`` — 1,054,330 variants in 625
blocks, of which 406 are dense (median 451 variants) and 219 are low-rank
(median 3,120 variants at median rank 890).

    python benchmarks/real_ld_gram.py --ld /path/to/ldpred3_ldref_hm3.npz

With ``--scores DIR`` the panel is built from real PGS Catalog scoring files
aligned to the reference by :func:`multipgs.align_to_reference`. Without it the
panel is synthetic — random variant supports and weights over the *real* LD —
which exercises the representation and cost questions but tells you nothing
about real score collinearity. Which panel was used is recorded in the
provenance, and the two must not be compared with each other.

Nothing here is an accuracy claim. It measures moments and their cost.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import multipgs
from multipgs.sumstat import _validate_moments, _weight_columns, score_gram


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _peak_rss_gb():
    """Peak resident set size of this process, in GB.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux; getting this wrong
    misreports memory by a factor of 1024, in the reassuring direction.
    """
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / 1024.0 ** 3 if sys.platform == "darwin" else peak / 1024.0 ** 2


def _sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _block_census(blocks):
    """Count blocks by representation, and their variant and rank totals."""
    from ldpred3 import LowRankLD

    from multipgs._ldpred3_compat import dequantize_ld

    census = {"n_blocks": 0, "n_lowrank": 0, "n_dense": 0,
              "variants_lowrank": 0, "variants_dense": 0, "rank_total": 0}
    for corr, idx in blocks:
        census["n_blocks"] += 1
        block = dequantize_ld(corr)
        if isinstance(block, LowRankLD):
            census["n_lowrank"] += 1
            census["variants_lowrank"] += int(idx.size)
            census["rank_total"] += int(np.asarray(block.U).shape[1])
        else:
            census["n_dense"] += 1
            census["variants_dense"] += int(idx.size)
    return census


def _synthetic_panel(n_variants, n_scores, per_score, seed):
    """A panel with realistic support sizes but no real score structure."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_scores):
        size = min(int(per_score), int(n_variants))
        idx = np.sort(rng.choice(n_variants, size=size, replace=False))
        # Effect sizes on the standardized-genotype scale of a polygenic score:
        # many small weights, so the Gram is dominated by LD, not by outliers.
        pairs.append((idx, rng.standard_normal(size) * 1e-3))
    ids = [f"synthetic_{i:04d}" for i in range(n_scores)]
    return pairs, ids, {"panel": "synthetic", "per_score": int(per_score)}


def _catalog_panel(directory, blocks_meta):
    """Align real PGS Catalog scoring files to the reference's variant table."""
    from multipgs import align_to_reference

    files = sorted(p for p in Path(directory).iterdir()
                   if p.name.endswith((".txt", ".txt.gz", ".tsv", ".tsv.gz")))
    if not files:
        raise SystemExit(f"no scoring files found in {directory}")
    pairs, ids, log = align_to_reference(
        [str(p) for p in files], blocks_meta["variants"],
        af=blocks_meta["af"], hwe_genotype_sd=True, on_error="skip")
    if not pairs:
        raise SystemExit("no scoring file aligned to the reference; check the "
                         "genome build")
    return pairs, ids, {"panel": "pgs_catalog", **log}


def _reference(path):
    """Load an ldpred3 LD cache plus the variant table needed for alignment."""
    from ldpred3 import load_ld_blocks

    started = time.perf_counter()
    blocks, variant_ids = load_ld_blocks(str(path))
    load_seconds = time.perf_counter() - started

    payload = np.load(str(path), allow_pickle=True)
    variants = {"id": np.asarray(payload["ids"], dtype=object),
                "chrom": np.asarray(payload["chrom"], dtype=object),
                "pos": np.asarray(payload["pos"], dtype=np.int64),
                "a1": np.asarray(payload["counted_allele"], dtype=object),
                "a2": np.asarray(payload["other_allele"], dtype=object)}
    af = (np.asarray(payload["reference_af"], dtype=float)
          if "reference_af" in payload else None)
    return (blocks, {"variants": variants, "af": af,
                     "n_variants": int(len(variant_ids)),
                     "load_seconds": load_seconds})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ld", required=True, type=Path,
                        help="ldpred3 LD cache (.npz) written by save_ld_blocks")
    parser.add_argument("--scores", type=Path, default=None,
                        help="directory of PGS Catalog scoring files; without "
                             "it the panel is synthetic over the real LD")
    parser.add_argument("--n-scores", type=int, default=200)
    parser.add_argument("--per-score", type=int, default=100_000,
                        help="synthetic panel only: variants per score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-lowrank", action="store_true",
                        help="also form the Gram with every block routed "
                             "through ld_matmul, and compare")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    blocks, meta = _reference(args.ld)
    census = _block_census(blocks)
    print(f"reference: {meta['n_variants']:,} variants, "
          f"{census['n_blocks']} blocks "
          f"({census['n_lowrank']} low-rank, {census['n_dense']} dense), "
          f"loaded in {meta['load_seconds']:.1f}s")

    if args.scores is not None:
        pairs, ids, panel_log = _catalog_panel(args.scores, meta)
    else:
        pairs, ids, panel_log = _synthetic_panel(
            meta["n_variants"], args.n_scores, args.per_score, args.seed)
    n_entries = int(sum(len(idx) for idx, _ in pairs))
    print(f"panel: {len(pairs)} scores, {n_entries:,} weight entries "
          f"({panel_log['panel']})")

    # A cache written with mmap=True returns block views into the file, so the
    # first pass over the reference pays to page in a gigabyte from disk. Timing
    # one route before the other then measures the page cache, not the
    # arithmetic — and flatters whichever route runs second by a factor of two.
    warm_start = time.perf_counter()
    warmed = 0
    for corr, _ in blocks:
        block = np.asarray(getattr(corr, "U", corr))
        # Touch one element per 4 KB page: enough to fault the payload in,
        # without the cost of reducing over a gigabyte.
        stride = max(1, 4096 // max(block.dtype.itemsize, 1))
        warmed += int(np.count_nonzero(block.reshape(-1)[::stride]))
    warm_seconds = time.perf_counter() - warm_start
    print(f"warmed {census['n_blocks']} blocks in {warm_seconds:.1f}s")

    t0 = time.perf_counter()
    gram, score_var = score_gram(pairs, blocks, n_variants=meta["n_variants"])
    gram_seconds = time.perf_counter() - t0
    peak_gb = _peak_rss_gb()
    print(f"score_gram: {gram_seconds:.1f}s, peak RSS {peak_gb:.1f} GB")

    lowrank_max_diff = None
    reference_seconds = None
    if args.check_lowrank:
        # Same Gram with every block forced through ld_matmul, which is what
        # the low-rank fast path has to reproduce.
        from ldpred3 import ld_matmul
        rows, cols, vals, m, k = _weight_columns(pairs, meta["n_variants"])
        order = np.argsort(rows, kind="stable")
        rows, cols, vals = rows[order], cols[order], vals[order]
        naive = np.zeros((k, k))
        t0 = time.perf_counter()
        for corr, idx in blocks:
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            lo_i, hi_i = np.searchsorted(rows, (lo, hi))
            if lo_i == hi_i:
                continue
            active = np.unique(cols[lo_i:hi_i])
            local_cols = np.searchsorted(active, cols[lo_i:hi_i])
            block_w = np.zeros((idx.size, active.size))
            block_w[rows[lo_i:hi_i] - lo, local_cols] = vals[lo_i:hi_i]
            naive[np.ix_(active, active)] += (
                block_w.T @ np.asarray(ld_matmul(corr, block_w), dtype=float))
        reference_seconds = time.perf_counter() - t0
        naive = 0.5 * (naive + naive.T)
        scale = max(float(np.max(np.abs(naive))), np.finfo(float).tiny)
        lowrank_max_diff = float(np.max(np.abs(gram - naive)) / scale)
        print(f"ld_matmul route: {reference_seconds:.1f}s, "
              f"max relative difference {lowrank_max_diff:.2e}, "
              f"speedup {reference_seconds / gram_seconds:.2f}x")

    # What the fitter's own validation sees. c is not available without a
    # target GWAS, so a zero cross-moment isolates the Gram's own properties.
    t0 = time.perf_counter()
    _, factor, coherence = _validate_moments(
        np.zeros(gram.shape[0]), gram, 1.0, label="real-LD panel")
    validate_seconds = time.perf_counter() - t0

    diagonal = np.diag(gram)
    active = diagonal > 0.0
    if np.any(active):
        sd = np.sqrt(diagonal[active])
        correlation = gram[np.ix_(active, active)] / np.outer(sd, sd)
        eigenvalues = np.linalg.eigvalsh(0.5 * (correlation + correlation.T))
        condition = (float(eigenvalues[-1] / eigenvalues[0])
                     if eigenvalues[0] > 0 else float("inf"))
        off = correlation[~np.eye(correlation.shape[0], dtype=bool)]
        max_abs_corr = float(np.max(np.abs(off))) if off.size else 0.0
    else:
        eigenvalues = np.zeros(0)
        condition, max_abs_corr = float("nan"), float("nan")

    row = {
        "panel": panel_log["panel"],
        "n_scores": int(gram.shape[0]),
        "n_variants_reference": meta["n_variants"],
        "n_weight_entries": n_entries,
        "n_blocks": census["n_blocks"],
        "n_blocks_lowrank": census["n_lowrank"],
        "n_blocks_dense": census["n_dense"],
        "variants_in_lowrank_blocks": census["variants_lowrank"],
        "mean_lowrank_rank": (census["rank_total"] / census["n_lowrank"]
                              if census["n_lowrank"] else float("nan")),
        "ld_load_seconds": meta["load_seconds"],
        "ld_warm_seconds": warm_seconds,
        "score_gram_seconds": gram_seconds,
        "ld_matmul_route_seconds": reference_seconds,
        "lowrank_max_relative_difference": lowrank_max_diff,
        "validate_moments_seconds": validate_seconds,
        "peak_rss_gb": peak_gb,
        "n_dead_scores": int(np.sum(~active)),
        "gram_rank": coherence["gram_rank"],
        "gram_min_correlation_eigenvalue":
            coherence["gram_min_correlation_eigenvalue"],
        "gram_psd_projected": coherence["gram_psd_projected"],
        "correlation_condition_number": condition,
        "max_abs_offdiagonal_correlation": max_abs_corr,
        "score_variance_min": float(np.min(diagonal)),
        "score_variance_median": float(np.median(diagonal)),
        "score_variance_max": float(np.max(diagonal)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "real_ld_gram.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    spectrum = args.output_dir / "real_ld_gram_spectrum.csv"
    with spectrum.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("index", "correlation_eigenvalue"))
        for i, value in enumerate(eigenvalues[::-1]):
            writer.writerow((i, float(value)))

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "command": [sys.executable, str(Path(__file__).resolve()),
                    *(sys.argv[1:] if argv is None else argv)],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "multipgs": multipgs.__version__,
        "numpy": np.__version__,
        "ldpred3": _version("ldpred3"),
        "numba": _version("numba"),
        "ld_reference": {"path": str(args.ld), "sha256": _sha256(args.ld)},
        "panel": panel_log,
        "parameters": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
        "note": ("Moments and their cost on a real LD reference; no accuracy "
                 "is estimated and none should be read into these numbers. A "
                 "synthetic panel shares the reference's block structure but "
                 "not real score collinearity, so synthetic and catalog rows "
                 "are not comparable."),
    }
    with (args.output_dir / "real_ld_gram_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(f"gram rank {row['gram_rank']} of {row['n_scores']}, "
          f"condition {condition:.3g}, "
          f"{row['n_dead_scores']} dead score(s)")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage-resolved cost of the summary-statistic pipeline at real dimensions.

``stack_scaling.py`` times the individual-level fit, but its defaults are a few
thousand simulated people and its own README says to supply representative
dimensions before making a deployment claim. Nobody has. The
summary-statistic route — the one that is supposed to scale to a catalog-sized
panel over a genome-wide reference — has had no cost benchmark at all, so the
only available answer to "can I run 900 scores over HapMap3 on this machine?"
has been a shrug.

This benchmark answers that question, and only that question. Holding a **real**
LD reference fixed, it sweeps ``K`` (component scores) and per-score variant
support, and reports wall time and absolute peak RSS separately for each stage
of :func:`multipgs.multi_pgs_sumstats`:

``panel``
    Building the weight panel. With ``--scores`` this is
    :func:`multipgs.align_to_reference` over real PGS Catalog scoring files —
    gzip parsing, rsID matching, allele harmonisation, HWE rescaling. Without
    it the panel is generated in memory, which is **not** a measurement of
    alignment cost and is reported under a different ``panel`` label so the two
    can never be confused.
``parse``
    :func:`multipgs.sumstat._weight_columns`, the sparse ``(index, weight)``
    parse. This is the memory wall, not the time wall.
``gram``
    :func:`multipgs.sumstat._score_gram_from_coo`, streaming the reference
    block by block. That is the private entry point, taking the already-parsed
    weights, because it is what the fitter calls; the public
    :func:`multipgs.sumstat.score_gram` is this plus the ``parse`` stage, and
    timing it here would count the parse twice.
``cross_moment``
    ``W_gwas^T z``. It parses the GWAS-basis weights a **second** time, so the
    pipeline holds two independent copies of the sparse panel at its peak even
    when ``weights_gwas is weights_ld``.
``validate``
    :func:`multipgs.sumstat._validate_moments`, an ``O(K^3)`` eigendecomposition
    of the ``K x K`` correlation matrix.
``path_residual``
    Total :func:`multipgs.multi_pgs_sumstats` time minus the four stages the
    fitter repeats internally — ``parse``, ``gram``, ``cross_moment`` and
    ``validate``. ``panel`` is **not** subtracted: the panel is built before
    the fitter is called and the fitter never rebuilds it. It is a
    **residual**, not a direct measurement, and the only stage here that is
    not measured directly: the
    path solve is welded into the fitter between a lambda grid, a boundedness
    certificate and a selection sweep, and re-implementing that seam in a
    benchmark would measure a copy that silently drifts from the real one. The
    residual therefore inherits any difference between running the repeated
    stages standalone and running them inside the fit. The standalone
    measurements go first, so they pay the cold cache and the fitter's repeat
    of them runs warmer; on a small or ``--max-blocks``-truncated reference
    that fits in last-level cache the difference can exceed the path cost and
    drive the residual **slightly negative**. That is the honest signal that
    the case is too small for the residual to mean anything, not a bug — and
    the row says so rather than leaving it to be inferred: ``residual_informative``
    is 0 exactly when the residual came out negative, in both CSVs. When it is
    0 the stage *fractions* in the summary are unusable (they sum above one
    and ``path_residual_fraction`` is negative); the directly measured
    ``*_seconds`` columns are unaffected. The residual also contains two
    further ``O(K^3)`` factorizations besides the path solve — the boundedness
    basis and, under ``tune="pumas"``, the PSD square root used for
    pseudo-splitting.

Splitting the stages is the point. They scale differently in ``K``: the Gram is
roughly linear in the number of non-zero weight entries and quadratic in the
scores active per block, while validation is cubic in ``K`` and completely
indifferent to how many variants each score carries. The crossover — where a
panel stops being Gram-bound and becomes eigendecomposition-bound — is the
result a user needs in order to predict their own run, and it is what the
``dominant_stage`` column reports.

**The memory model.** The sparse parse materializes three arrays over every
non-zero entry: ``int64`` variant index, ``int64`` score column, ``float64``
value, so 24 bytes per entry, measured and reported as
``bytes_per_nonzero``. An equivalent dense ``float32`` ``(m, K)`` matrix costs
4 bytes per *cell*. Sparse is therefore cheaper only below a density of
``4/24 = 1/6``: on a 1,054,330-variant reference that is a per-score support of
about 175,700 variants. Sparse catalog scores sit far below it — 900 scores at
5,000 variants each is 108 MB and nobody notices. A panel of genome-wide dense
HapMap3 scores sits at density 1, where the same 900 scores cost 21.2 GiB
sparse against 3.5 GiB dense, doubled to 42 GiB because ``weights_ld`` and
``weights_gwas`` are parsed separately and coexist. That is the real constraint
on a catalog-scale dense panel, and it is why the parse is the memory wall.

A second, LD-free worker materializes both representations and checks the
arithmetic rather than asserting it. It also exposes the catch:
``_weight_columns`` given a dense matrix immediately calls ``np.nonzero`` and
builds the very same COO arrays, so today a dense panel handed to multipgs
costs dense storage *plus* the sparse parse. The crossover says which
representation a future dense Gram path ought to consume; it is not a saving
that is available now.

**What this benchmark cannot establish.** It measures cost, and nothing else.

* **No accuracy claim of any kind.** No held-out R², no calibration, no
  comparison of tuning rules. Nothing here says the fitted combination is any
  good. ``fit_accuracy.py``, ``sumstat_vs_individual.py`` and
  ``sumstat_calibration.py`` are where accuracy lives.
* The timed fit uses ``tune="pumas"`` by default, which is **regime B** —
  pseudo-validation on a split of the single available GWAS, tuning and not
  clean assessment. ``--tune none`` is **regime C**, fitted and selected on the
  same unsplit GWAS and optimistically biased. The label is recorded as
  ``timed_tune_regime`` and not as a bare ``regime`` column for a reason: it
  selects *which code path is being timed* and nothing more. No number in the
  output CSV is a regime-A, regime-B or regime-C accuracy estimate, because no
  number in the output CSV is an accuracy estimate. In particular ``fit`` is
  never evaluated: no ``pseudo_r2``, no ``selection_mse``, no
  ``evaluate_sumstat`` call appears anywhere here, precisely because under both
  timed settings the only moments available are the ones that fitted and
  selected the model. ``tune="independent"`` is not
  timed because it needs a second reference; from the source it adds one
  further ``parse`` plus ``gram`` pass over the tuning weights, so its extra
  cost is the measured ``parse`` and ``gram`` columns again.
* Unless ``--scores`` is given, the panel is synthetic over the real LD:
  random variant supports and small random weights. That reproduces the
  reference's block structure and the panel's *shape*, which is what the parse
  and Gram costs depend on, but not real score collinearity. Coordinate descent
  converges faster on a near-orthogonal panel than on the near-duplicate
  catalog scores multi-PGS actually combines, so ``path_residual_seconds`` from
  a synthetic panel is a **lower bound** on the real path cost, and synthetic
  and catalog rows must not be compared with each other.
* ``z`` is synthetic in every configuration, rescaled so the strongest single
  score has plug-in marginal correlation ``--target-marginal-r`` with the
  trait. That is there to make the path solve do representative work. No
  quantity derived from it — ``n_selected`` included — means anything
  biological; it is reported so a reader can confirm the path was not trivially
  empty.
* Peak RSS is the worker process's absolute high-water mark and includes the
  Python, NumPy and Numba runtime plus the ~1.6 GB of the loaded reference. The
  per-stage ``rss_gb_after_*`` columns are cumulative high-water marks, so
  their differences are incremental, not independent. Compare only rows
  produced by the same command in the same environment.
* ``--max-blocks`` truncates the reference to its first blocks. It exists for
  smoke tests. Rows produced with it are **not** real-dimension measurements
  and record the truncation in the provenance.
* One machine, one BLAS, one run per case. There is no repetition and no
  variance estimate, so small differences between adjacent cases are noise.

The reference is supplied, not shipped. Developed against Privé's bigsnpr
HapMap3+ European (UK Biobank) LD converted to ldpred3 block format by
``ldpred3/benchmarks/convert_bigsnpr_ldref.py``: 1,054,330 variants in 625
blocks, 406 dense and 219 low-rank, ``n_ref=362,320``.

Full-scale run — the whole 1,054,330-variant reference, six cases, roughly
20 minutes and up to about 5 GB resident on an M-series laptop. This is the
command to quote in a deployment claim, and it is deliberately not the smoke
test::

    python benchmarks/sumstat_cost.py \\
        --ld /Users/au507860/REPOS/ldpred3/benchmarks/.work/ldref-hm3/ldpred3_ldref_hm3.npz \\
        --cases 100x5000 300x5000 900x5000 300x200000 3000x30 3000x5

The first three sweep ``K`` at a support large enough that every score touches
every large LD block; ``300x200000`` moves the support axis instead, where the
Gram barely notices and the parse grows fortyfold; the last two bracket the
crossover, where the ``O(K^3)`` validation overtakes the Gram. Push past it
with ``--cases 6000x10`` if you want to watch validation win outright — expect
several more gigabytes for the eigendecomposition's workspace.

Smoke test, on a truncated reference, which is a proof that the script runs and
nothing else::

    python benchmarks/sumstat_cost.py --ld .../ldpred3_ldref_hm3.npz \\
        --max-blocks 40 --n-lambda 20 --cases 20x500 60x20000 400x50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Pinned before NumPy is imported, in the parent and in every worker that
# re-imports this module. One thread per process keeps rows comparable — a
# BLAS that silently grabs every core turns an O(K^3) column into a
# measurement of the machine's idleness — and leaves the machine usable while
# a long sweep runs. The parent passes an explicit --blas-threads through the
# worker environment, where setdefault then leaves it alone.
for _threads in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import numpy as np

# Measure the checkout beside this artifact even when no editable install is
# active, and make the sibling benchmarks importable so this script reuses
# stack_scaling's process accounting instead of growing a second copy of it.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import multipgs
from multipgs.sumstat import (_score_cross_moment, _score_gram_from_coo,
                              _validate_moments, _weight_columns)
from stack_scaling import _rss_mb, _version


# The stages competing for dominance. path_residual is derived, not measured,
# which is why it is last: a reader scanning the row meets the direct
# measurements first.
STAGES = ("panel", "parse", "gram", "cross_moment", "validate",
          "path_residual")

# int64 index + int64 score column + float64 value, versus one float32 cell.
SPARSE_BYTES_PER_ENTRY = 24
DENSE_BYTES_PER_CELL = 4


def _peak_rss_gb():
    """Absolute peak RSS of this process in GB, via stack_scaling's accounting.

    That helper already resolves the macOS-bytes / Linux-kibibytes trap, and
    getting it wrong misreports memory by a factor of 1024 in the reassuring
    direction.
    """
    return _rss_mb() / 1024.0


def _sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_case(case):
    """``"900x5000"`` to ``(900, 5000)``, matching stack_scaling's case syntax."""
    try:
        first, second = case.lower().split("x", 1)
        k, support = int(first), int(second)
    except ValueError:
        raise SystemExit(f"case {case!r} must look like KxSUPPORT, e.g. "
                         "900x5000") from None
    if k < 2 or support < 1:
        raise SystemExit(f"case {case!r} needs at least 2 scores and 1 variant "
                         "of support")
    return k, support


def _reference(path, max_blocks, need_variants):
    """Load an ldpred3 LD cache, optionally truncated, plus its variant table.

    Truncation keeps a prefix of the block list, which still tiles
    ``0..m-1`` exactly once — the layout ``score_gram`` validates — so a
    truncated reference is a smaller but structurally valid one.

    The variant table is read only when a real scoring file has to be aligned
    against it. It is a million-element object array, and materializing it in a
    run that never uses it would put a hundred megabytes of irrelevance into
    every peak-RSS column on this page.

    It is built as ldpred3's own :class:`~ldpred3.genotype_model.VariantTable`
    and not as a plain mapping. ``align_to_reference`` reads only ``id`` from
    it directly and would accept a dict, but it hands the table down to
    ``ldpred3.harmonize``, which reads every field by attribute. A dict
    therefore fails inside the per-file ``try`` and, under the ``on_error``
    policy this benchmark uses, is swallowed into a skipped file rather than
    raised — every score fails to align for a reason that looks like a build
    mismatch. ``ld_reference_shrinkage.py`` documents the same trap.
    """
    from ldpred3 import load_ld_blocks

    started = time.perf_counter()
    blocks, variant_ids = load_ld_blocks(str(path))
    load_seconds = time.perf_counter() - started

    if max_blocks is not None and max_blocks < len(blocks):
        blocks = list(blocks[:max_blocks])
    n_variants = int(sum(int(idx.size) for _, idx in blocks))

    variants = af = None
    if need_variants:
        from ldpred3.genotype_model import VariantTable

        payload = np.load(str(path), allow_pickle=True)
        variants = VariantTable(
            id=np.asarray(payload["ids"])[:n_variants].astype(str),
            chrom=np.asarray(payload["chrom"])[:n_variants].astype(str),
            pos=np.asarray(payload["pos"], dtype=np.int64)[:n_variants],
            cm=np.zeros(n_variants),
            a1=np.asarray(payload["counted_allele"])[:n_variants].astype(str),
            a2=np.asarray(payload["other_allele"])[:n_variants].astype(str))
        if "reference_af" in payload:
            af = np.asarray(payload["reference_af"], dtype=float)[:n_variants]
    meta = {"variants": variants, "af": af, "n_variants": n_variants,
            "n_variants_full": int(len(variant_ids)),
            "load_seconds": load_seconds}
    return blocks, meta


def _block_census(blocks):
    """Count blocks by representation; the two kinds cost very different Grams."""
    from ldpred3 import LowRankLD

    from multipgs._ldpred3_compat import dequantize_ld

    census = {"n_blocks": 0, "n_blocks_lowrank": 0, "n_blocks_dense": 0}
    for corr, _ in blocks:
        census["n_blocks"] += 1
        if isinstance(dequantize_ld(corr), LowRankLD):
            census["n_blocks_lowrank"] += 1
        else:
            census["n_blocks_dense"] += 1
    return census


def _synthetic_pairs(n_variants, n_scores, per_score, seed):
    """A panel with realistic shape and no real score structure.

    Weights are small and many, so the Gram is dominated by LD rather than by
    a handful of outliers; the parse and Gram costs depend on the support
    pattern, which is what this reproduces, and not on the values.
    """
    rng = np.random.default_rng(seed)
    support = min(int(per_score), int(n_variants))
    pairs = []
    for _ in range(n_scores):
        idx = np.sort(rng.choice(n_variants, size=support, replace=False))
        pairs.append((idx, rng.standard_normal(support) * 1e-3))
    ids = [f"synthetic_{i:04d}" for i in range(n_scores)]
    return pairs, ids


def _catalog_pairs(directory, meta, n_scores):
    """Align the first ``n_scores`` real scoring files to the reference.

    ``on_error="skip"`` drops a file that fails to align rather than raising,
    so ``K`` can come out below the case's request. That is recorded twice
    over: the returned log carries ``n_requested``, ``n_aligned``, ``n_failed``
    and the per-file error into the provenance, and the row's ``k`` column is
    the number that actually aligned, never the number asked for.
    """
    from multipgs import align_to_reference

    files = sorted(p for p in Path(directory).iterdir()
                   if p.name.endswith((".txt", ".txt.gz", ".tsv", ".tsv.gz")))
    if len(files) < n_scores:
        raise SystemExit(f"{directory} holds {len(files)} scoring files but the "
                         f"case asks for {n_scores}")
    requested = files[:n_scores]
    pairs, ids, log = align_to_reference(
        [str(p) for p in requested], meta["variants"], af=meta["af"],
        hwe_genotype_sd=True, on_error="skip")
    log = {"panel": "pgs_catalog", "files": [p.name for p in requested], **log}
    if len(pairs) < 2:
        # Quote what actually failed. "Check the genome build" is one cause
        # among several and on its own sends a reader to the wrong place: a
        # --max-blocks reference covers only the first blocks of the genome,
        # against which a genome-wide score legitimately aligns to nothing.
        raise SystemExit(
            f"only {len(pairs)} of {len(requested)} scoring files aligned to "
            f"this reference's {meta['n_variants']:,} variants "
            f"(of {meta['n_variants_full']:,} in the untruncated cache). "
            f"Per-file errors: {log.get('errors', 'none reported')}")
    return pairs, ids, log


def _synthetic_z(pairs, n_variants, seed, n_signal):
    """Marginal effects that make the path solve do representative work.

    A pure-noise ``z`` selects nothing and would time an empty path. Loading a
    few of the panel's own weight vectors into ``z`` gives the lasso something
    to find. The overall scale is fixed later against the Gram diagonal, once
    that exists.
    """
    rng = np.random.default_rng(seed + 977)
    z = rng.standard_normal(n_variants) * 1e-4
    for j in rng.choice(len(pairs), size=min(n_signal, len(pairs)),
                        replace=False):
        idx, weight = pairs[j]
        z[idx] += weight
    return z


def _warm(blocks, n_variants, seed):
    """Compile and page in everything the timed stages will touch.

    Two distinct kinds of cold start would otherwise land in the first case's
    numbers: ldpred3's Numba kernels and multipgs' coordinate descent compile
    on first call, and a memory-mapped cache faults its payload in on first
    touch. Both are paid here.
    """
    from multipgs import multi_pgs_sumstats

    warm_start = time.perf_counter()
    for corr, _ in blocks:
        block = np.asarray(getattr(corr, "U", corr))
        # One element per 4 KB page: enough to fault the payload in without
        # paying to reduce over a gigabyte of it.
        stride = max(1, 4096 // max(block.dtype.itemsize, 1))
        int(np.count_nonzero(block.reshape(-1)[::stride]))
    warm_seconds = time.perf_counter() - warm_start

    tiny_pairs, _ = _synthetic_pairs(n_variants, 6, min(200, n_variants), seed)
    tiny_z = _synthetic_z(tiny_pairs, n_variants, seed, 2)
    multi_pgs_sumstats(tiny_pairs, tiny_z, blocks, weights_gwas=tiny_pairs,
                       n_variants_ld=n_variants, n_lambda=5, tune="none")
    return warm_seconds


def _pipeline_worker(args):
    """One (K, support) case end to end, in its own process."""
    from multipgs import multi_pgs_sumstats

    k, support = _parse_case(args.worker_pipeline)
    blocks, meta = _reference(args.ld, args.max_blocks,
                              args.scores is not None)
    m = meta["n_variants"]
    census = _block_census(blocks)
    rss_after_load = _peak_rss_gb()
    warm_seconds = _warm(blocks, m, args.seed)

    panel_started = time.perf_counter()
    if args.scores is not None:
        pairs, score_ids, panel_log = _catalog_pairs(args.scores, meta, k)
        panel = "pgs_catalog"
    else:
        pairs, score_ids = _synthetic_pairs(m, k, support, args.seed)
        # Support is clamped to the reference size, which a --max-blocks run
        # can easily trip. Record what was asked for and what was realized, so
        # the clamp is visible in the provenance and not only inferable from
        # the row's support_* columns.
        panel_log = {"panel": "synthetic", "per_score_requested": support,
                     "per_score_realized": min(support, m),
                     "clamped_to_reference": int(support > m)}
        panel = "synthetic"
    panel_seconds = time.perf_counter() - panel_started
    rss_after_panel = _peak_rss_gb()

    k = len(pairs)
    supports = np.asarray([idx.size for idx, _ in pairs], dtype=np.int64)
    z = _synthetic_z(pairs, m, args.seed, args.signal_scores)

    # The four stages the fitter performs internally — parse, gram,
    # cross_moment, validate — measured one at a time on exactly the inputs it
    # will use, and the four that path_residual subtracts. Each artifact is
    # sized and then released before the full fit runs, so the fit's peak
    # reflects one pipeline's working set rather than two.
    t0 = time.perf_counter()
    parsed = _weight_columns(pairs, m)
    parse_seconds = time.perf_counter() - t0
    rows, cols, vals = parsed[0], parsed[1], parsed[2]
    n_entries = int(vals.size)
    sparse_bytes = int(rows.nbytes + cols.nbytes + vals.nbytes)
    rss_after_parse = _peak_rss_gb()

    t0 = time.perf_counter()
    gram, score_var = _score_gram_from_coo(parsed, blocks)
    gram_seconds = time.perf_counter() - t0
    rss_after_gram = _peak_rss_gb()

    t0 = time.perf_counter()
    c, _, _ = _score_cross_moment(pairs, z, k, "weights_gwas")
    cross_moment_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    _validate_moments(c, gram, args.var_y, label="cost benchmark")
    validate_seconds = time.perf_counter() - t0
    rss_after_validate = _peak_rss_gb()

    # Fix z's scale against the Gram now that it exists: the strongest single
    # score gets the requested plug-in marginal correlation with the trait.
    # Rescaling z is O(m) and does not disturb any stage timing above.
    sd = np.sqrt(np.maximum(score_var, 0.0))
    marginal = np.abs(c) / np.where(sd > 0.0, sd, np.inf)
    strongest = float(np.max(marginal)) if marginal.size else 0.0
    if strongest > 0.0:
        z = z * (args.target_marginal_r / strongest)

    del parsed, rows, cols, vals, gram, score_var, c

    fit_kwargs = {"tune": args.tune}
    if args.tune == "pumas":
        fit_kwargs |= {"n_eff": float(args.n_eff),
                       "weights_independent_of_z": True,
                       "n_repeats": args.n_repeats}
    t0 = time.perf_counter()
    fit = multi_pgs_sumstats(pairs, z, blocks, weights_gwas=pairs,
                             score_ids=score_ids, n_variants_ld=m,
                             var_y=args.var_y, n_lambda=args.n_lambda,
                             rng=args.seed if args.tune == "pumas" else None,
                             **fit_kwargs)
    sumstats_total_seconds = time.perf_counter() - t0
    rss_after_fit = _peak_rss_gb()

    measured = {"panel": panel_seconds, "parse": parse_seconds,
                "gram": gram_seconds, "cross_moment": cross_moment_seconds,
                "validate": validate_seconds}
    measured["path_residual"] = sumstats_total_seconds - (
        parse_seconds + gram_seconds + cross_moment_seconds + validate_seconds)

    return {
        "case": args.worker_pipeline,
        "panel": panel,
        "tune": args.tune,
        # Named for what it is: the evidence class the timed code path *would*
        # carry if it produced an accuracy estimate. It does not. A bare
        # "regime" column would read as an evidence class attached to a number
        # in this table, and every number in this table is a cost.
        "timed_tune_regime": "B" if args.tune == "pumas" else "C",
        "k": int(k),
        "per_score_requested": support,
        "support_median": int(np.median(supports)),
        "support_min": int(supports.min()),
        "support_max": int(supports.max()),
        "n_variants_reference": m,
        "n_variants_full_reference": meta["n_variants_full"],
        "n_weight_entries": n_entries,
        "density": n_entries / float(m * k),
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "n_blocks": census["n_blocks"],
        "n_blocks_lowrank": census["n_blocks_lowrank"],
        "n_blocks_dense": census["n_blocks_dense"],
        "ld_load_seconds": meta["load_seconds"],
        "ld_warm_seconds": warm_seconds,
        "panel_seconds": panel_seconds,
        "parse_seconds": parse_seconds,
        "gram_seconds": gram_seconds,
        "cross_moment_seconds": cross_moment_seconds,
        "validate_seconds": validate_seconds,
        "sumstats_total_seconds": sumstats_total_seconds,
        "path_residual_seconds": measured["path_residual"],
        # A negative residual means the standalone stages, which run first and
        # pay the cold cache, cost more alone than the whole fit that repeats
        # them warm. The case is then too small for the residual to carry
        # information, and every stage fraction derived from it is unusable.
        # Flagged in the row so a reader of the CSV alone can see it without
        # having to notice a fraction above one.
        "residual_informative": int(measured["path_residual"] > 0.0),
        "pipeline_seconds": panel_seconds + sumstats_total_seconds,
        "dominant_stage": max(STAGES, key=lambda name: measured[name]),
        "sparse_parse_bytes": sparse_bytes,
        "bytes_per_nonzero": (sparse_bytes / n_entries if n_entries
                              else float("nan")),
        "rss_gb_after_load": rss_after_load,
        "rss_gb_after_panel": rss_after_panel,
        "rss_gb_after_parse": rss_after_parse,
        "rss_gb_after_gram": rss_after_gram,
        "rss_gb_after_validate": rss_after_validate,
        "rss_gb_after_fit": rss_after_fit,
        "peak_rss_gb": _peak_rss_gb(),
        # Loading the reference allocates transient decompression buffers whose
        # size wanders between runs, so peak_rss_gb alone hides the part that
        # actually scales with K and support. This difference is that part —
        # and a floor, because a pipeline whose working set never exceeds the
        # loader's transient peak reports zero here.
        "rss_gb_over_load": _peak_rss_gb() - rss_after_load,
        "n_selected": int(fit.n_selected),
        "n_path_points": int(fit.path.shape[0]),
        "panel_log": panel_log,
    }


def _memory_worker(args):
    """Sparse against dense storage for one panel shape, with no LD loaded.

    This runs in its own process precisely so that allocating a dense
    ``(m, K)`` matrix does not contaminate the pipeline row's peak RSS, and so
    that the LD reference — 1.6 GB that has nothing to do with the question —
    is absent from both numbers.
    """
    k, support = _parse_case(args.worker_memory)
    m = int(args.n_variants)

    # A first allocation in a fresh process pays first-touch page faults that
    # have nothing to do with the encoding under test. Spend them here, on a
    # panel small enough to be free, so the two parses below are compared on
    # equal terms.
    _weight_columns(_synthetic_pairs(min(m, 4096), 4, 8, args.seed)[0],
                    min(m, 4096))
    baseline = _peak_rss_gb()

    pairs, _ = _synthetic_pairs(m, k, support, args.seed)
    t0 = time.perf_counter()
    parsed = _weight_columns(pairs, m)
    sparse_parse_seconds = time.perf_counter() - t0
    sparse_bytes = int(parsed[0].nbytes + parsed[1].nbytes + parsed[2].nbytes)
    n_entries = int(parsed[2].size)
    rss_after_sparse = _peak_rss_gb()
    del parsed, pairs

    dense_bytes = DENSE_BYTES_PER_CELL * m * k
    row = {
        "memory_case": args.worker_memory,
        "memory_n_variants": m,
        "sparse_measured_bytes": sparse_bytes,
        "memory_sparse_parse_seconds": sparse_parse_seconds,
        "dense_float32_model_bytes": int(dense_bytes),
        "sparse_over_dense": sparse_bytes / float(dense_bytes),
        # The crossover is a property of the two encodings alone: 24 bytes per
        # entry against 4 bytes per cell, so sparse wins below one-sixth
        # density, i.e. below m/6 variants of support per score.
        "crossover_density": DENSE_BYTES_PER_CELL / SPARSE_BYTES_PER_ENTRY,
        "crossover_support_variants": m * DENSE_BYTES_PER_CELL
                                      / SPARSE_BYTES_PER_ENTRY,
        "dense_alloc_bytes": None,
        "memory_dense_parse_seconds": None,
        "dense_parse_coo_bytes": None,
        "rss_gb_memory_baseline": baseline,
        "rss_gb_after_sparse_parse": rss_after_sparse,
        "rss_gb_after_dense_parse": None,
        "dense_measured": 0,
    }

    # Materializing the dense form costs the dense bytes plus, because
    # _weight_columns immediately calls np.nonzero on it, the very same COO
    # arrays the sparse route builds. Measuring it is therefore only affordable
    # under an explicit budget — and the result is the honest half of the
    # memory story: today a dense panel handed to multipgs is strictly more
    # expensive than a sparse one, whatever the encoding arithmetic says.
    if dense_bytes + SPARSE_BYTES_PER_ENTRY * n_entries <= args.dense_max_gb * 1024 ** 3:
        pairs, _ = _synthetic_pairs(m, k, support, args.seed)
        dense = np.zeros((m, k), dtype=np.float32)
        for j, (idx, weight) in enumerate(pairs):
            dense[idx, j] = weight
        del pairs
        t0 = time.perf_counter()
        parsed = _weight_columns(dense)
        dense_parse_seconds = time.perf_counter() - t0
        row |= {
            "dense_alloc_bytes": int(dense.nbytes),
            "memory_dense_parse_seconds": dense_parse_seconds,
            "dense_parse_coo_bytes": int(parsed[0].nbytes + parsed[1].nbytes
                                         + parsed[2].nbytes),
            "rss_gb_after_dense_parse": _peak_rss_gb(),
            "dense_measured": 1,
        }
    return row


def _run(script, mode, case, args, extra=()):
    command = [sys.executable, str(script), mode, case,
               "--ld", str(args.ld),
               "--seed", str(args.seed),
               "--tune", args.tune,
               "--n-eff", str(args.n_eff),
               "--n-repeats", str(args.n_repeats),
               "--n-lambda", str(args.n_lambda),
               "--var-y", str(args.var_y),
               "--signal-scores", str(args.signal_scores),
               "--target-marginal-r", str(args.target_marginal_r),
               "--dense-max-gb", str(args.dense_max_gb),
               *extra]
    if args.max_blocks is not None:
        command += ["--max-blocks", str(args.max_blocks)]
    if args.scores is not None:
        command += ["--scores", str(args.scores)]
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(args.blas_threads)
    # capture_output keeps the worker's diagnostics out of the parent's stdout,
    # where they would corrupt the JSON payload — but CalledProcessError does
    # not carry stderr into its message, so a worker that dies would otherwise
    # report only an exit status. Re-raise with the worker's own words.
    proc = subprocess.run(command, capture_output=True, text=True,
                          env=environment)
    if proc.returncode != 0:
        raise SystemExit(f"worker for case {case} exited {proc.returncode}:\n"
                         f"{proc.stderr.strip() or proc.stdout.strip()}")
    # ldpred3 and NumPy occasionally write advisories to stdout; the payload is
    # always the final line, so parse that rather than the whole stream.
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"worker for case {case} produced no output:\n"
                         f"{proc.stderr}")
    return json.loads(lines[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ld", required=True, type=Path,
                        help="ldpred3 LD cache (.npz) written by save_ld_blocks")
    parser.add_argument("--cases", nargs="+",
                        default=("100x5000", "300x5000", "900x5000",
                                 "300x200000", "3000x30", "3000x5"),
                        help="KxSUPPORT per case: component scores by variants "
                             "per score")
    parser.add_argument("--scores", type=Path, default=None,
                        help="directory of PGS Catalog scoring files; each "
                             "case then aligns its first K files and the "
                             "case's support is ignored")
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="truncate the reference to its first N blocks; "
                             "for smoke tests only, not a real-dimension run")
    parser.add_argument("--tune", choices=("pumas", "none"), default="pumas",
                        help="pumas is regime B, none is regime C; this "
                             "selects the timed code path, not an accuracy "
                             "claim")
    parser.add_argument("--n-eff", type=float, default=300_000.0)
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--n-lambda", type=int, default=100)
    parser.add_argument("--var-y", type=float, default=1.0)
    parser.add_argument("--signal-scores", type=int, default=5)
    parser.add_argument("--target-marginal-r", type=float, default=0.3)
    parser.add_argument("--dense-max-gb", type=float, default=3.0,
                        help="budget for materializing the dense (m, K) "
                             "comparison; a case over it reports the modelled "
                             "size only")
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-memory-model", action="store_true",
                        help="omit the sparse-versus-dense worker")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--worker-pipeline", help=argparse.SUPPRESS)
    parser.add_argument("--worker-memory", help=argparse.SUPPRESS)
    parser.add_argument("--n-variants", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_pipeline:
        print(json.dumps(_pipeline_worker(args)))
        return 0
    if args.worker_memory:
        print(json.dumps(_memory_worker(args)))
        return 0

    if min(args.n_repeats, args.n_lambda, args.signal_scores) < 1 \
            or args.blas_threads < 1 or args.n_eff <= 0.0 \
            or not 0.0 < args.target_marginal_r < 1.0 \
            or (args.max_blocks is not None and args.max_blocks < 1):
        parser.error("numeric arguments are too small for this design")

    script = Path(__file__).resolve()
    started = time.perf_counter()
    rows, panel_logs = [], {}
    for case in args.cases:
        pipeline = _run(script, "--worker-pipeline", case, args)
        # The alignment log belongs in the provenance, not in a CSV column: for
        # a catalog panel it names every scoring file that failed to align.
        panel_logs[case] = pipeline.pop("panel_log")
        print(f"{case}: {pipeline['k']} scores, "
              f"{pipeline['n_weight_entries']:,} weight entries over "
              f"{pipeline['n_variants_reference']:,} variants -> "
              f"{pipeline['pipeline_seconds']:.1f}s, peak "
              f"{pipeline['peak_rss_gb']:.2f} GB, dominated by "
              f"{pipeline['dominant_stage']}")
        if args.skip_memory_model:
            memory = {}
        else:
            # The memory question is about panel shape alone, so the worker is
            # given the entries-per-score this case actually produced. That
            # makes the model row valid for a catalog panel of uneven support
            # too, where the case string's support is meaningless.
            per_score = max(1, round(pipeline["n_weight_entries"]
                                     / pipeline["k"]))
            memory = _run(script, "--worker-memory",
                          f"{pipeline['k']}x{per_score}", args,
                          extra=["--n-variants",
                                 str(pipeline["n_variants_reference"])])
            print(f"    sparse {memory['sparse_measured_bytes'] / 1024 ** 3:.2f} GB "
                  f"vs dense float32 "
                  f"{memory['dense_float32_model_bytes'] / 1024 ** 3:.2f} GB "
                  f"({memory['sparse_over_dense']:.2f}x)")
        rows.append(pipeline | memory)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "sumstat_cost.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # The summary is the table a user reads to predict their own run: which
    # share of the wall clock each stage took, which one won, and how far the
    # sparse encoding is from the dense crossover.
    summary = []
    for row in rows:
        total = max(row["pipeline_seconds"], np.finfo(float).tiny)
        item = {"case": row["case"], "panel": row["panel"], "k": row["k"],
                "n_weight_entries": row["n_weight_entries"],
                "density": row["density"],
                "pipeline_seconds": row["pipeline_seconds"]}
        for stage in STAGES:
            item[f"{stage}_fraction"] = row[f"{stage}_seconds"] / total
        item |= {"dominant_stage": row["dominant_stage"],
                 # Carried into the summary because this is the table that
                 # invites arithmetic. When it is 0 the residual came out
                 # negative, so path_residual_fraction is negative and the
                 # other fractions sum above one; the case is too small to
                 # inform the residual and none of these fractions mean
                 # anything. The directly measured *_seconds columns in the
                 # raw CSV are unaffected.
                 "residual_informative": row["residual_informative"],
                 "peak_rss_gb": row["peak_rss_gb"],
                 "rss_gb_over_load": row["rss_gb_over_load"],
                 "sparse_parse_gb": row["sparse_parse_bytes"] / 1024 ** 3,
                 "bytes_per_nonzero": row["bytes_per_nonzero"],
                 # "model", not a measurement: the dense figure is the encoding
                 # arithmetic 4 * m * K. dense_measured says whether a dense
                 # matrix was actually materialized to check it against.
                 "dense_float32_model_gb":
                     (row["dense_float32_model_bytes"] / 1024 ** 3
                      if "dense_float32_model_bytes" in row else None),
                 "dense_measured": row.get("dense_measured"),
                 "sparse_over_dense": row.get("sparse_over_dense"),
                 "crossover_support_variants":
                     row.get("crossover_support_variants")}
        summary.append(item)
    summary_path = args.output_dir / "sumstat_cost_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "command": [sys.executable, str(script),
                    *(sys.argv[1:] if argv is None else argv)],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "multipgs": multipgs.__version__,
        "numpy": np.__version__,
        "ldpred3": _version("ldpred3"),
        "numba": _version("numba"),
        "cpu_count": os.cpu_count(),
        "blas_threads": args.blas_threads,
        "ld_reference": {"path": str(args.ld), "sha256": _sha256(args.ld),
                         "truncated_to_blocks": args.max_blocks},
        "panel_source": ("pgs_catalog" if args.scores is not None
                         else "synthetic"),
        "panel_log": panel_logs,
        "parameters": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
        "note": ("Cost only: no accuracy is estimated here and none should be "
                 "read into any column. tune='pumas' is regime B and "
                 "tune='none' is regime C, recorded as timed_tune_regime "
                 "rather than as a bare 'regime' column because that label "
                 "selects which code path was timed, not the provenance of an "
                 "accuracy number; no pseudo_r2, selection_mse or "
                 "evaluate_sumstat result is produced, because under both "
                 "timed settings the only available moments are the ones that "
                 "fitted and selected the model. "
                 "path_residual_seconds is total multi_pgs_sumstats time minus "
                 "the four separately measured stages the fitter repeats "
                 "(parse, gram, cross_moment, validate; panel is built outside "
                 "the fitter and is not subtracted), so it is "
                 "a residual and also contains the boundedness basis and, "
                 "under PUMAS, the pseudo-split PSD factor — two further "
                 "O(K^3) terms besides the path solve. The standalone stage "
                 "measurements run first and pay the cold cache, so on a small "
                 "or truncated reference the residual can be slightly "
                 "negative; that means the case is too small for the residual "
                 "to be informative and is flagged as residual_informative=0, "
                 "which also invalidates every stage fraction in the summary "
                 "for that row. A synthetic panel has "
                 "the real reference's block structure and the panel's shape "
                 "but not real score collinearity, so its path cost is a lower "
                 "bound and synthetic and catalog rows are not comparable. z "
                 "is synthetic in every configuration. Peak RSS is absolute "
                 "worker-process RSS including the Python, NumPy and Numba "
                 "runtime and the loaded reference; per-stage rss_gb_after_* "
                 "are cumulative high-water marks, so only rows from the same "
                 "command and environment may be compared. A run with "
                 "--max-blocks is not a real-dimension measurement."),
    }
    with (args.output_dir / "sumstat_cost_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Streaming score Gram ``W^T D W`` from an LD reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validate import _positive_integer


def _integer_indices(values, label):
    """Return integer-valued indices without truncating malformed input."""
    raw = np.asarray(values)
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain integer variant indices")
    flat = raw.ravel()
    if raw.dtype.kind == "f":
        limit = np.iinfo(np.int64).max
        if (not np.all(np.isfinite(flat))
                or np.any(flat != np.floor(flat))
                or np.any(flat < -limit - 1) or np.any(flat > limit)):
            raise ValueError(f"{label} must contain integer variant indices")
    elif raw.dtype.kind == "u" and np.any(flat > np.iinfo(np.int64).max):
        raise ValueError(f"{label} contains an index outside int64 range")
    return flat.astype(np.int64, copy=False)


def _nonnegative_integer(value, label):
    """Validate a count before converting it to an integer."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a non-negative integer")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{label} must be a non-negative integer")
    number = float(raw)
    if (not np.isfinite(number) or number < 0.0 or not number.is_integer()
            or number > np.iinfo(np.int64).max):
        raise ValueError(f"{label} must be a non-negative integer")
    return int(number)


def _as_blocks(ld, n_variants):
    """Normalize an LD argument to ``(corr_block, idx)`` pairs.

    Accepts ldpred3's native block list, a single dense correlation matrix, or
    anything ``ldpred3.ld_matmul`` understands paired with explicit indices.
    Non-dense inputs are consumed lazily and must tile ``0..m-1`` exactly once.
    """
    if isinstance(ld, np.ndarray):
        if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
            raise ValueError(f"a dense LD matrix must be square, got {ld.shape}")
        if ld.shape[0] != n_variants:
            raise ValueError(f"LD matrix is {ld.shape[0]} x {ld.shape[0]} but "
                             f"the weights cover {n_variants} variants")
        return [(ld, np.arange(n_variants))]

    def validated():
        expected = 0
        seen = False
        for b, block in enumerate(ld):
            if not (isinstance(block, tuple) and len(block) == 2):
                raise ValueError(
                    f"LD block {b} is not a (corr_block, idx) pair; this is the "
                    "layout ldpred3.compute_ld_blocks and "
                    "ldpred3.load_ld_blocks return, got "
                    f"{type(block).__name__}")
            corr, idx = block
            idx = _integer_indices(idx, f"LD block {b} indices")
            if idx.size == 0:
                continue
            seen = True
            if np.any(np.diff(idx) != 1):
                raise ValueError(
                    f"LD block {b} does not cover a contiguous run of variants. "
                    "The streaming Gram slices weights by block range; use "
                    "ldpred3.compute_ld_blocks, whose blocks tile 0..m-1.")
            if int(idx[0]) != expected:
                kind = "overlaps an earlier block" if int(idx[0]) < expected \
                    else "leaves a gap before it"
                raise ValueError(
                    f"LD block {b} starts at variant {int(idx[0])}, expected "
                    f"{expected}; it {kind}. Blocks must tile 0..m-1 exactly "
                    "once and in order.")
            expected = int(idx[-1]) + 1
            yield corr, idx
        if not seen:
            raise ValueError("the LD reference has no blocks")
        if expected != n_variants:
            raise ValueError(f"the LD blocks cover variants 0..{expected - 1} "
                             f"but the weights cover 0..{n_variants - 1}")

    return validated()


@dataclass(frozen=True)
class _DenseWeights:
    """Validated dense weights kept dense throughout the moment calculation."""

    values: np.ndarray

    @property
    def m(self):
        return int(self.values.shape[0])

    @property
    def k(self):
        return int(self.values.shape[1])


def _dense_weights_finite(weights):
    """Check finite values with at most about one million Boolean temporaries."""
    rows = max(1, 1_000_000 // max(int(weights.shape[1]), 1))
    return all(np.all(np.isfinite(weights[start:start + rows]))
               for start in range(0, weights.shape[0], rows))


def _weight_columns(weights, n_variants=None):
    """Normalize weights to a dense view or canonical sparse COO tuple.

    Dense ``(m, K)`` input remains dense.  Turning it into COO first costs three
    extra arrays per non-zero (24 bytes with int64 indices and float64 values),
    which is catastrophic precisely when a matrix is dense. A panel of catalog
    scores is usually sparse, so a list of per-score ``(index, weight)`` pairs
    remains the representation that scales for that use case.
    """
    if isinstance(weights, np.ndarray):
        if weights.ndim != 2:
            raise ValueError(f"dense weights must be 2-D (m, K), got "
                             f"{weights.shape}")
        if not _dense_weights_finite(weights):
            raise ValueError("dense weights contain non-finite values")
        m, k = weights.shape
        if k == 0:
            raise ValueError("dense weights contain no score columns")
        if n_variants is not None:
            requested = _nonnegative_integer(n_variants, "n_variants")
            if requested != m:
                raise ValueError(
                    f"n_variants={requested} but dense weights have {m} rows")
        values = np.asarray(weights)
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(float)
        return _DenseWeights(values)

    pairs = list(weights)
    if not pairs:
        raise ValueError("no score weights given")
    idx_parts, col_parts, val_parts = [], [], []
    largest = -1
    for k, pair in enumerate(pairs):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError(f"score {k} must be an (index, weight) pair, got "
                             f"{type(pair).__name__}")
        idx = _integer_indices(pair[0], f"score {k} indices")
        val = np.asarray(pair[1], dtype=float).ravel()
        if idx.shape != val.shape:
            raise ValueError(f"score {k} has {idx.size} indices and {val.size} "
                             "weights")
        if idx.size and not np.all(np.isfinite(val)):
            raise ValueError(f"score {k} has non-finite weights")
        if idx.size:
            if int(idx.min()) < 0:
                raise ValueError(f"score {k} has a negative variant index")
            # A sparse score is a mathematical vector, not an insertion log.
            # Coalescing here gives W'DW and W'z the same interpretation.
            unique, inverse = np.unique(idx, return_inverse=True)
            if unique.size != idx.size:
                combined = np.zeros(unique.size, dtype=float)
                np.add.at(combined, inverse, val)
                keep = combined != 0.0
                idx, val = unique[keep], combined[keep]
            else:
                # Sparse input is a mathematical vector, so an explicitly
                # stored zero is not an entry and must not alter memory logs or
                # the inferred reference extent.
                keep = val != 0.0
                idx, val = idx[keep], val[keep]
            if idx.size:
                largest = max(largest, int(idx.max()))
        idx_parts.append(idx)
        col_parts.append(np.full(idx.size, k, dtype=np.int64))
        val_parts.append(val)

    m = (int(largest + 1) if n_variants is None else
         _nonnegative_integer(n_variants, "n_variants"))
    if m <= largest:
        raise ValueError(f"a weight indexes variant {largest} but the reference "
                         f"has {m}")
    return (np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64),
            np.concatenate(col_parts) if col_parts else np.zeros(0, np.int64),
            np.concatenate(val_parts) if val_parts else np.zeros(0),
            m, len(pairs))


def score_gram(weights, ld, *, n_variants=None):
    """The ``K x K`` score covariance ``W^T D W`` from an LD reference.

    Streams the reference one LD block at a time, densifying only that block's
    slice of the weights, so peak memory is ``O(block_size * K)`` rather than
    ``O(m * K)``. For a 900-score panel and 500-variant blocks that is a few
    megabytes instead of tens of gigabytes.

    Parameters
    ----------
    weights : ndarray or sequence of (index, weight)
        Per-variant weights for each score, on the **standardized** genotype
        scale, aligned to the LD reference's variants. PGS Catalog weights count
        raw alleles and must be converted first — see :func:`align_to_reference`.
    ld : sequence of (corr_block, idx), or ndarray
        ldpred3 LD blocks, or one dense correlation matrix.
    n_variants : int, optional
        Reference size, needed only when the weights are sparse and no score
        touches the last variant.

    Returns
    -------
    (gram, score_var) : (ndarray, ndarray)
        ``gram`` is ``W^T D W`` (``K x K``); ``score_var`` is its diagonal, the
        variance of each score under the reference's LD.
    """
    return _score_gram_from_coo(_weight_columns(weights, n_variants), ld)


def _block_quadform(corr, block_w):
    """``W_b^T D_b W_b`` for one LD block, using its own representation.

    ldpred3 stores a large block as a low-rank factor (LR8):
    ``D = U U^T + diag(residual)``. Going through :func:`ldpred3.ld_matmul`
    computes ``U (U^T W)`` — projecting back up to the block's full variant
    dimension — only for this function to immediately contract it back down
    again. Keeping the factor instead,

        W^T D W = (U^T W)^T (U^T W) + (residual * W)^T W,

    skips that back-projection and shrinks the second product from ``O(k A^2)``
    to ``O(r A^2)``. In the bigsnpr HapMap3+ reference the low-rank blocks hold
    the bulk of the variants — median 3,120 variants at median rank 890, so
    ``r/k`` is about 0.29 — and this is where a genome-wide Gram spends its
    time.

    LDpred3 owns bounded decoding and factor contractions for every supported
    representation, so compact storage is not followed by whole-block widening.
    """
    from ldpred3.interop import ld_crossproducts

    return ld_crossproducts(corr, block_w)


def _score_gram_from_coo(parsed, ld):
    """:func:`score_gram` on already-parsed dense or sparse weights.

    Parsing a sparse weight set materializes three arrays over every non-zero
    entry, which for a genome-wide panel is the largest allocation in the whole
    fit. A caller that already holds the parse passes it here instead of
    handing the raw weights back to be parsed a second time.
    """
    if isinstance(parsed, _DenseWeights):
        matrix = parsed.values
        blocks = _as_blocks(ld, parsed.m)
        gram = np.zeros((parsed.k, parsed.k), dtype=float)
        for corr, idx in blocks:
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            gram += _block_quadform(corr, matrix[lo:hi])
        gram = 0.5 * (gram + gram.T)
        return gram, np.diag(gram).copy()

    rows, cols, vals, m, k = parsed
    blocks = _as_blocks(ld, m)

    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]

    gram = np.zeros((k, k), dtype=float)
    for corr, idx in blocks:
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        start, stop = np.searchsorted(rows, (lo, hi))
        if start == stop:
            continue
        # Catalog scores are sparse across both variants and blocks. Work only
        # on the scores touching this block; forming a B x K matrix and a full
        # K x K product here can waste two orders of magnitude of work.
        active = np.unique(cols[start:stop])
        local_cols = np.searchsorted(active, cols[start:stop])
        block_w = np.zeros((idx.size, active.size), dtype=float)
        block_w[rows[start:stop] - lo, local_cols] = vals[start:stop]
        gram[np.ix_(active, active)] += _block_quadform(corr, block_w)

    # W^T D W is symmetric in exact arithmetic; the accumulation is not, and an
    # asymmetric Gram makes the coordinate descent's covariance updates drift.
    gram = 0.5 * (gram + gram.T)
    return gram, np.diag(gram).copy()


def _weight_digest(rows, cols, vals, n_variants, n_scores):
    """Canonical digest of the exact aligned score matrix used by a fit."""
    import hashlib

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    keep = vals != 0.0
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    order = np.lexsort((cols, rows))
    digest = hashlib.sha256()
    digest.update(np.asarray([n_variants, n_scores], dtype="<i8").tobytes())
    digest.update(rows[order].astype("<i8", copy=False).tobytes())
    digest.update(cols[order].astype("<i8", copy=False).tobytes())
    digest.update(vals[order].astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _parsed_weight_info(parsed):
    """Return ``(m, K, number of non-zero entries)`` without expanding dense W."""
    if isinstance(parsed, _DenseWeights):
        return parsed.m, parsed.k, int(np.count_nonzero(parsed.values))
    _, _, vals, m, k = parsed
    return int(m), int(k), int(vals.size)


def _parsed_weight_digest(parsed):
    """Canonical sparse-matrix digest without a dense-to-COO allocation.

    The byte stream matches :func:`_weight_digest`: non-zero row indices, then
    column indices, then values, all in row-major order. Dense inputs are walked
    in bounded row chunks and never materialize genome-wide coordinate arrays.
    """
    if not isinstance(parsed, _DenseWeights):
        return _weight_digest(*parsed)

    import hashlib

    matrix = parsed.values
    digest = hashlib.sha256()
    digest.update(np.asarray([parsed.m, parsed.k], dtype="<i8").tobytes())
    chunk_rows = max(1, min(parsed.m, 1_000_000 // max(parsed.k, 1)))
    for part in ("row", "col", "value"):
        for start in range(0, parsed.m, chunk_rows):
            block = matrix[start:start + chunk_rows]
            row, col = np.nonzero(block)
            if part == "row":
                value = row.astype(np.int64, copy=False) + start
                digest.update(value.astype("<i8", copy=False).tobytes())
            elif part == "col":
                digest.update(col.astype("<i8", copy=False).tobytes())
            else:
                value = block[row, col]
                digest.update(value.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _collapse_parsed_weights(parsed, coefficients):
    """Collapse component scores without changing the stored weight basis."""
    coefficients = np.asarray(coefficients, dtype=float)
    if isinstance(parsed, _DenseWeights):
        return parsed.values @ coefficients
    rows, cols, vals, m, _ = parsed
    out = np.zeros(m, dtype=float)
    np.add.at(out, rows, vals * coefficients[cols])
    return out


def _ld_variant_count(weights_ld, ld, explicit, label):
    """Infer one LD source's variant count without borrowing a GWAS length."""
    if explicit is not None:
        count = _positive_integer(explicit, label)
        if isinstance(weights_ld, np.ndarray):
            if weights_ld.ndim != 2:
                raise ValueError(f"{label} weights must be two-dimensional")
            if weights_ld.shape[0] != count:
                raise ValueError(
                    f"{label}={count} but dense LD weights have "
                    f"{weights_ld.shape[0]} rows")
        if isinstance(ld, np.ndarray):
            if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
                raise ValueError(f"{label} LD matrix must be square")
            if ld.shape[0] != count:
                raise ValueError(
                    f"{label}={count} but dense LD has {ld.shape[0]} rows")
        return count
    if isinstance(weights_ld, np.ndarray):
        if weights_ld.ndim != 2:
            raise ValueError(f"{label} weights must be two-dimensional")
        return int(weights_ld.shape[0])
    if isinstance(ld, np.ndarray):
        if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
            raise ValueError(f"{label} LD matrix must be square")
        return int(ld.shape[0])
    if isinstance(ld, (list, tuple)) and ld:
        last = _integer_indices(ld[-1][1], f"{label} LD block indices")
        if last.size:
            return int(last[-1]) + 1
    return None


def _score_cross_moment(weights_gwas, z, n_scores, label):
    """Compute ``W_gwas' z`` on that GWAS's own standardized-genotype basis."""
    parsed = _weight_columns(weights_gwas, int(z.size))
    return _score_cross_moment_parsed(parsed, z, n_scores, label)


def _score_cross_moment_parsed(parsed, z, n_scores, label, *, n_entries=None):
    """Cross-moment from an existing parse, preserving one-shot iterables."""
    if isinstance(parsed, _DenseWeights):
        m, k = parsed.m, parsed.k
        if m != z.size:
            raise ValueError(f"{label} weights cover {m} variants but z covers "
                             f"{z.size}")
        if k != n_scores:
            raise ValueError(f"{label} weights describe {k} scores but the LD "
                             f"weights describe {n_scores}; score identity and "
                             "column order must agree")
        if n_entries is None:
            n_entries = int(np.count_nonzero(parsed.values))
        return parsed.values.T @ z, n_entries, m

    rows, cols, vals, m, k = parsed
    if m != z.size:
        raise ValueError(f"{label} weights cover {m} variants but z covers "
                         f"{z.size}")
    if k != n_scores:
        raise ValueError(f"{label} weights describe {k} scores but the LD "
                         f"weights describe {n_scores}; score identity and "
                         "column order must agree")
    c = np.zeros(k, dtype=float)
    np.add.at(c, cols, vals * z[rows])
    return c, int(vals.size) if n_entries is None else int(n_entries), m


def score_moments(weights_ld, z, ld, *, weights_gwas=None,
                  n_variants_ld=None):
    """The score-space moments ``(c, G)`` for one set of summary statistics.

    The pair that :func:`evaluate_sumstat` scores against, and the same pair
    :func:`multi_pgs_sumstats` fits from. Building them for an *evaluation* GWAS
    is how a combination gets an honest regime A number. ``weights_gwas`` and
    ``weights_ld`` represent the same raw component scores, but each is
    multiplied by the empirical genotype SD of its own dataset:
    ``c = W_gwas.T @ z`` and ``G = W_ld.T @ D @ W_ld``. They may cover different
    variant sets; only their score columns must agree. Equality with
    individual-level Gaussian regression moments is exact for unadjusted data,
    or when genotypes and phenotype were jointly residualized on the identical
    covariate design—not for arbitrary adjusted marginal GWAS coefficients.
    """
    if weights_gwas is None:
        raise ValueError(
            "weights_gwas is required separately from weights_ld; pass the "
            "same matrix explicitly only when GWAS and LD genotype scales are "
            "genuinely identical")
    z = np.asarray(z, dtype=float).ravel()
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values")
    n_variants_ld = _ld_variant_count(
        weights_ld, ld, n_variants_ld, "n_variants_ld")
    parsed = _weight_columns(weights_ld, n_variants_ld)
    gram, var = _score_gram_from_coo(parsed, ld)
    if weights_gwas is weights_ld:
        c, _, _ = _score_cross_moment_parsed(
            parsed, z, gram.shape[0], "weights_gwas")
    else:
        c, _, _ = _score_cross_moment(
            weights_gwas, z, gram.shape[0], "weights_gwas")
    return c, gram, var
